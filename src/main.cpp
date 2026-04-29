#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iomanip>
#include "stb_image.h"
#include "stb_image_write.h"
#include "dither.h"
#include "ola_espline.h"

namespace fs = std::filesystem;

extern void scaleFSR_EASU(const unsigned char* input, int inW, int inH,
                           unsigned char* output, int outW, int outH,
                           float lfga, bool tepd);
extern void applyFSR_RCAS(const unsigned char* input, int w, int h,
                           unsigned char* output, float sharpness,
                           bool useDenoise, float lfga, bool tepd);
extern void scaleNearestNeighbor(const unsigned char* input, int inW, int inH,
                                  unsigned char* output, int outW, int outH);
extern void scaleBilinear(const unsigned char* input, int inW, int inH,
                           unsigned char* output, int outW, int outH,
                           float lfga, bool tepd);
extern void scaleBicubic(const unsigned char* input, int inW, int inH,
                          unsigned char* output, int outW, int outH,
                          float lfga, bool tepd);
extern void scaleLanczos3(const unsigned char* input, int inW, int inH,
                           unsigned char* output, int outW, int outH,
                           float lfga, bool tepd);

// ---------------------------------------------------------------------------
// PSNR
// ---------------------------------------------------------------------------
double calculatePSNR(const unsigned char* img1, const unsigned char* img2,
                     int width, int height)
{
    double mse = 0.0;
    for (int i = 0; i < width * height * 4; i += 4) {
        double dr = (double)img1[i+0] - (double)img2[i+0];
        double dg = (double)img1[i+1] - (double)img2[i+1];
        double db = (double)img1[i+2] - (double)img2[i+2];
        mse += dr*dr + dg*dg + db*db;
    }
    mse /= (width * height * 3.0);
    if (mse <= 1e-7) return 100.0;
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

// ---------------------------------------------------------------------------
// AUTO-TUNER
// ---------------------------------------------------------------------------
void optimizeRCAS(const unsigned char* originalImg, int width, int height,
                  const std::string& upAlgo, float downScale,
                  const std::string& downAlgo, bool rcasDenoise,
                  bool& outUseRcas, float& outSharpness)
{
    std::cout << "  [Auto-Tuner] Running PSNR optimization pass ("
              << downScale << "x " << downAlgo
              << " downscale -> " << upAlgo << " upscale)..." << std::endl;

    int downW = std::max(1, (int)(width  * downScale));
    int downH = std::max(1, (int)(height * downScale));

    unsigned char* downData = new unsigned char[downW * downH * 4];
    unsigned char* upData   = new unsigned char[width  * height * 4];

    if      (downAlgo == "nearest")  scaleNearestNeighbor(originalImg, width, height, downData, downW, downH);
    else if (downAlgo == "bilinear") scaleBilinear(originalImg, width, height, downData, downW, downH, 0.f, false);
    else if (downAlgo == "lanczos3") scaleLanczos3(originalImg, width, height, downData, downW, downH, 0.f, false);
    else                             scaleBicubic (originalImg, width, height, downData, downW, downH, 0.f, false);

    if      (upAlgo == "nearest")  scaleNearestNeighbor(downData, downW, downH, upData, width, height);
    else if (upAlgo == "bilinear") scaleBilinear(downData, downW, downH, upData, width, height, 0.f, false);
    else if (upAlgo == "bicubic")  scaleBicubic (downData, downW, downH, upData, width, height, 0.f, false);
    else if (upAlgo == "lanczos3") scaleLanczos3(downData, downW, downH, upData, width, height, 0.f, false);
    else if (upAlgo == "fsr")      scaleFSR_EASU(downData, downW, downH, upData, width, height, 0.f, false);
    // OLA e-spline is not included in auto-tuner (it has its own internal optimisation)

    double bestPsnr  = calculatePSNR(originalImg, upData, width, height);
    outUseRcas    = false;
    outSharpness  = 0.f;

    unsigned char* rcasData = new unsigned char[width * height * 4];
    for (int i = 0; i <= 20; ++i) {
        float testSharpness = i / 10.f;
        applyFSR_RCAS(upData, width, height, rcasData,
                      testSharpness, false, 0.f, false);
        double psnr = calculatePSNR(originalImg, rcasData, width, height);
        if (psnr > bestPsnr) {
            bestPsnr     = psnr;
            outUseRcas   = true;
            outSharpness = testSharpness;
        }
    }

    std::cout << "  [Auto-Tuner] Peak PSNR: "
              << std::fixed << std::setprecision(2) << bestPsnr << " dB -> ";
    if (outUseRcas)
        std::cout << "Winning Setting: RCAS ON (Sharpness " << outSharpness << ")" << std::endl;
    else
        std::cout << "Winning Setting: RCAS OFF" << std::endl;

    delete[] downData;
    delete[] upData;
    delete[] rcasData;
}

// ---------------------------------------------------------------------------
// PROCESS ONE IMAGE
// ---------------------------------------------------------------------------
bool processImage(const std::string& inFile, const std::string& outFile,
                  float scale, const std::string& algo,
                  bool useRcas, float sharpness, bool rcasDenoise,
                  float lfga, bool tepd, int bpp,
                  const std::string& paletteMatch,
                  const std::string& paletteDither,
                  const std::string& matchPaletteFrom,
                  bool runPsnrOpt, float psnrDownScale,
                  const std::string& psnrDownAlgo,
                  const OLAESplineParams& olaParams)
{
    // ---- Palette handling (8-bit output) -----------------------------------
    std::vector<ColorRGBA> targetPalette;
    bool hasTransparency = false;
    bool shouldMatch     = false;

    if (bpp == 8) {
        if (!matchPaletteFrom.empty()) {
            if (!loadOriginalPalette(matchPaletteFrom, targetPalette, hasTransparency)) {
                std::cout << "Error: Could not load palette from "
                          << matchPaletteFrom << std::endl;
                return false;
            }
            shouldMatch = true;
        } else {
            bool is8Bit = loadOriginalPalette(inFile, targetPalette, hasTransparency);
            if      (paletteMatch == "on")   { if (!is8Bit) { std::cout << "Error: --palette-match on requires 8-bit input.\n"; return false; } shouldMatch = true; }
            else if (paletteMatch == "auto") { shouldMatch = is8Bit; }
            else                             { shouldMatch = false;   }
        }
    }

    // ---- Load source image -------------------------------------------------
    int width, height, channels;
    unsigned char* imgData = stbi_load(inFile.c_str(), &width, &height, &channels, 4);
    if (!imgData) {
        std::cout << "Error: Could not load " << inFile << std::endl;
        return false;
    }

    // ---- Auto-tuner --------------------------------------------------------
    if (runPsnrOpt && algo != "off" && algo != "ola") {
        optimizeRCAS(imgData, width, height, algo, psnrDownScale,
                     psnrDownAlgo, rcasDenoise, useRcas, sharpness);
    }

    // ---- Compute output dimensions -----------------------------------------
    int newW = width, newH = height;
    if (algo != "off") {
        if (algo == "ola") {
            // OLA e-spline requires integer scale factor
            int sf = olaParams.scaleFactor;
            newW = width  * sf;
            newH = height * sf;
        } else {
            newW = std::max(1, (int)(width  * scale));
            newH = std::max(1, (int)(height * scale));
        }
    }

    // ---- Scale -------------------------------------------------------------
    unsigned char* step1Data = new unsigned char[newW * newH * 4]();
    float pass1_lfga = useRcas ? 0.f : lfga;
    bool  pass1_tepd = useRcas ? false : tepd;

    if (algo == "off") {
        std::memcpy(step1Data, imgData, newW * newH * 4);
    } else if (algo == "nearest") {
        scaleNearestNeighbor(imgData, width, height, step1Data, newW, newH);
    } else if (algo == "bilinear") {
        scaleBilinear(imgData, width, height, step1Data, newW, newH,
                      pass1_lfga, pass1_tepd);
    } else if (algo == "bicubic") {
        scaleBicubic(imgData, width, height, step1Data, newW, newH,
                     pass1_lfga, pass1_tepd);
    } else if (algo == "lanczos3") {
        scaleLanczos3(imgData, width, height, step1Data, newW, newH,
                      pass1_lfga, pass1_tepd);
    } else if (algo == "fsr") {
        if (scale < 1.f) {
            std::cout << "Error: FSR cannot downscale." << std::endl;
            stbi_image_free(imgData); delete[] step1Data; return false;
        }
        scaleFSR_EASU(imgData, width, height, step1Data, newW, newH,
                      pass1_lfga, pass1_tepd);
    } else if (algo == "ola") {
        std::cout << "  [OLA e-spline] Pre-processing + B-spline + edge expansion..." << std::endl;
        std::cout << "  [OLA e-spline] CS optimisation may take a moment." << std::endl;
        scaleOLAESpline(imgData, width, height, step1Data, newW, newH, olaParams);
    }

    // ---- RCAS pass (not applicable after OLA — it has its own sharpening) --
    unsigned char* finalData = step1Data;
    if (useRcas && algo != "ola") {
        finalData = new unsigned char[newW * newH * 4];
        applyFSR_RCAS(step1Data, newW, newH, finalData,
                      sharpness, rcasDenoise, lfga, tepd);
        delete[] step1Data;
    }

    // ---- Save --------------------------------------------------------------
    if (bpp == 8) {
        if (!shouldMatch)
            generatePalette(finalData, newW, newH, targetPalette, hasTransparency);
        unsigned char* indexedData = new unsigned char[newW * newH];
        bool useFS = (paletteDither == "fs");
        quantizeAndDither(finalData, newW, newH, indexedData,
                          targetPalette, hasTransparency, useFS);
        saveIndexedPNG(outFile.c_str(), indexedData, newW, newH, targetPalette);
        delete[] indexedData;
    } else {
        int outChannels = 4;
        unsigned char* saveBuffer = finalData;
        if (bpp == 24) {
            outChannels = 3;
            saveBuffer  = new unsigned char[newW * newH * 3];
            for (int i = 0; i < newW * newH; ++i) {
                saveBuffer[i*3+0] = finalData[i*4+0];
                saveBuffer[i*3+1] = finalData[i*4+1];
                saveBuffer[i*3+2] = finalData[i*4+2];
            }
        }
        stbi_write_png(outFile.c_str(), newW, newH, outChannels,
                       saveBuffer, newW * outChannels);
        if (bpp == 24) delete[] saveBuffer;
    }

    stbi_image_free(imgData);
    delete[] finalData;
    return true;
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::cout << "--- Image Resizer CPU ---" << std::endl;

    std::string inputPath, outputPath, suffix;
    std::string algo       = "fsr";
    std::string rcasInput  = "auto";
    std::string paletteMatch     = "auto";
    std::string matchPaletteFrom = "";
    std::string paletteDither    = "none";

    float scale     = 2.f;
    float sharpness = 0.2f;
    float lfga      = 0.f;
    bool  rcasDenoise = false;
    bool  tepd        = false;
    int   bpp         = 32;

    // Auto-tuner
    bool        runPsnrOpt   = false;
    float       psnrDownScale = 0.5f;
    std::string psnrDownAlgo  = "bicubic";

    // OLA e-spline parameters (exposed as CLI options)
    OLAESplineParams olaParams;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if      (arg == "--scale"        && i+1<argc) scale        = std::stof(argv[++i]);
        else if (arg == "--algo"         && i+1<argc) algo         = argv[++i];
        else if (arg == "--sharpness"    && i+1<argc) sharpness    = std::stof(argv[++i]);
        else if (arg == "--suffix"       && i+1<argc) suffix       = argv[++i];
        else if (arg == "--rcas"         && i+1<argc) rcasInput    = argv[++i];
        else if (arg == "--rcas-denoise" && i+1<argc) rcasDenoise  = (std::string(argv[++i]) == "on");
        else if (arg == "--lfga"         && i+1<argc) lfga         = std::stof(argv[++i]);
        else if (arg == "--tepd"         && i+1<argc) tepd         = (std::string(argv[++i]) == "on");
        else if (arg == "--bpp"          && i+1<argc) bpp          = std::stoi(argv[++i]);
        else if (arg == "--palette-match"      && i+1<argc) paletteMatch      = argv[++i];
        else if (arg == "--match-palette-from" && i+1<argc) matchPaletteFrom  = argv[++i];
        else if (arg == "--palette-dither"     && i+1<argc) paletteDither     = argv[++i];

        // OLA e-spline options
        else if (arg == "--ola-scale"       && i+1<argc) olaParams.scaleFactor   = std::stoi(argv[++i]);
        else if (arg == "--ola-cs-nests"    && i+1<argc) olaParams.csNests       = std::stoi(argv[++i]);
        else if (arg == "--ola-cs-gen"      && i+1<argc) olaParams.csMaxGen      = std::stoi(argv[++i]);
        else if (arg == "--ola-cs-step"     && i+1<argc) olaParams.csFitnessStep = std::stoi(argv[++i]);
        else if (arg == "--ola-canny-low"   && i+1<argc) olaParams.cannyLow      = std::stof(argv[++i]);
        else if (arg == "--ola-canny-high"  && i+1<argc) olaParams.cannyHigh     = std::stof(argv[++i]);

        else if (arg == "--rcas-max-psnr") {
            runPsnrOpt = true;
            if (i+1 < argc) {
                std::string next = argv[i+1];
                if (next.find("--") != 0) {
                    size_t comma = next.find(',');
                    if (comma != std::string::npos) {
                        try {
                            psnrDownScale = std::stof(next.substr(0, comma));
                            psnrDownAlgo  = next.substr(comma + 1);
                        } catch(...) {}
                    }
                    ++i;
                }
            }
        }
        else if (inputPath.empty())  inputPath  = arg;
        else if (outputPath.empty()) outputPath = arg;
    }

    if (inputPath.empty() || outputPath.empty()) {
        std::cout << "Usage: image-resizer <in> <out> [options]\n"
                  << "\nScaling:\n"
                  << "  --scale 2.0\n"
                  << "  --algo fsr|lanczos3|bicubic|bilinear|nearest|ola|off\n"
                  << "\nOLA e-spline options (--algo ola):\n"
                  << "  --ola-scale 2          Integer scale factor (2 or 4)\n"
                  << "  --ola-cs-nests 25      CS population size\n"
                  << "  --ola-cs-gen 50        CS max generations\n"
                  << "  --ola-cs-step 4        Fitness pixel step (1=exact,4=fast)\n"
                  << "  --ola-canny-low 20     Canny low threshold\n"
                  << "  --ola-canny-high 50    Canny high threshold\n"
                  << "\nPost-FX:\n"
                  << "  --rcas on|off --sharpness 0.2 --rcas-denoise on|off\n"
                  << "  --lfga 0.0 --tepd on|off\n"
                  << "\nOutput:\n"
                  << "  --bpp 32|24|8 --suffix _hd\n"
                  << "\n8-Bit palette:\n"
                  << "  --palette-match auto|on|off\n"
                  << "  --match-palette-from <file>\n"
                  << "  --palette-dither fs|none\n"
                  << "\nAuto-Tune:\n"
                  << "  --rcas-max-psnr [0.5,bicubic]\n";
        return 1;
    }

    // When using OLA, synchronise the integer scale factor from --ola-scale
    // (or from --scale if --ola-scale was not given explicitly).
    if (algo == "ola") {
        // If user set --scale but not --ola-scale, derive it.
        // olaParams.scaleFactor default is 2; honour --ola-scale if given.
        // We leave scaleFactor as set by --ola-scale (default 2).
        (void)scale; // scale not used for ola path
    }

    bool useRcas = (rcasInput == "on" || (rcasInput == "auto" && algo == "fsr"));

    // ---- Process file(s) ---------------------------------------------------
    if (fs::is_directory(inputPath)) {
        if (!fs::exists(outputPath)) fs::create_directories(outputPath);
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            if (entry.is_regular_file() &&
                entry.path().extension().string() == ".png")
            {
                std::string outFilePath =
                    (fs::path(outputPath) /
                     (entry.path().stem().string() + suffix + ".png")).string();
                std::cout << " -> " << entry.path().filename().string() << std::endl;
                processImage(entry.path().string(), outFilePath,
                             scale, algo, useRcas, sharpness, rcasDenoise,
                             lfga, tepd, bpp, paletteMatch, paletteDither,
                             matchPaletteFrom, runPsnrOpt, psnrDownScale,
                             psnrDownAlgo, olaParams);
            }
        }
    } else {
        processImage(inputPath, outputPath,
                     scale, algo, useRcas, sharpness, rcasDenoise,
                     lfga, tepd, bpp, paletteMatch, paletteDither,
                     matchPaletteFrom, runPsnrOpt, psnrDownScale,
                     psnrDownAlgo, olaParams);
    }

    return 0;
}
