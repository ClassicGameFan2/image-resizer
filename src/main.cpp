#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <sstream>
#include "stb_image.h"
#include "stb_image_write.h"
#include "dither.h"
#include "metrics.h"

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
extern void scaleICBI(const unsigned char* input, int inW, int inH,
                       unsigned char* output, int outW, int outH,
                       int sz, int pf, bool vr, int st,
                       double tm, double tc, int sc, double ts,
                       double al, double bt, double gm,
                       bool fcbiOnly);

// =========================================================================
// PSNR (used by auto-tuner only)
// =========================================================================
static double calculatePSNR(const unsigned char* img1, const unsigned char* img2,
                             int width, int height) {
    double mse = 0.0;
    for (int i = 0; i < width * height; ++i) {
        for (int c = 0; c < 3; ++c) {
            double d = static_cast<double>(img1[i*4+c]) - static_cast<double>(img2[i*4+c]);
            mse += d * d;
        }
    }
    mse /= (width * height * 3.0);
    if (mse <= 1e-10) return 100.0;
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

// =========================================================================
// Auto-tuner
// =========================================================================
static void optimizeRCAS(const unsigned char* originalImg, int width, int height,
                          const std::string& upAlgo,
                          float downScale, const std::string& downAlgo,
                          bool rcasDenoise,
                          bool& outUseRcas, float& outSharpness,
                          // ICBI params (forwarded if upAlgo==icbi)
                          int icbi_sz, int icbi_pf, int icbi_st, double icbi_tm,
                          double icbi_tc, int icbi_sc, double icbi_ts,
                          double icbi_al, double icbi_bt, double icbi_gm,
                          bool icbi_fcbi) {

    std::cout << "  [Auto-Tuner] Running PSNR optimization pass..." << std::endl;

    int downW = std::max(1, (int)(width  * downScale));
    int downH = std::max(1, (int)(height * downScale));

    unsigned char* downData = new unsigned char[downW * downH * 4];
    unsigned char* upData   = new unsigned char[width  * height * 4];

    // Downscale
    if      (downAlgo == "nearest")  scaleNearestNeighbor(originalImg, width, height, downData, downW, downH);
    else if (downAlgo == "bilinear") scaleBilinear(originalImg, width, height, downData, downW, downH, 0.0f, false);
    else if (downAlgo == "lanczos3") scaleLanczos3(originalImg, width, height, downData, downW, downH, 0.0f, false);
    else                             scaleBicubic(originalImg, width, height, downData, downW, downH, 0.0f, false);

    // Upscale
    if      (upAlgo == "nearest")  scaleNearestNeighbor(downData, downW, downH, upData, width, height);
    else if (upAlgo == "bilinear") scaleBilinear(downData, downW, downH, upData, width, height, 0.0f, false);
    else if (upAlgo == "bicubic")  scaleBicubic(downData, downW, downH, upData, width, height, 0.0f, false);
    else if (upAlgo == "lanczos3") scaleLanczos3(downData, downW, downH, upData, width, height, 0.0f, false);
    else if (upAlgo == "fsr")      scaleFSR_EASU(downData, downW, downH, upData, width, height, 0.0f, false);
    else if (upAlgo == "icbi")     scaleICBI(downData, downW, downH, upData, width, height,
                                              icbi_sz, icbi_pf, false, icbi_st,
                                              icbi_tm, icbi_tc, icbi_sc, icbi_ts,
                                              icbi_al, icbi_bt, icbi_gm, icbi_fcbi);

    double bestPsnr = calculatePSNR(originalImg, upData, width, height);
    outUseRcas  = false;
    outSharpness = 0.0f;

    unsigned char* rcasData = new unsigned char[width * height * 4];
    for (int i = 0; i <= 20; ++i) {
        float testSharpness = i / 10.0f;
        applyFSR_RCAS(upData, width, height, rcasData, testSharpness, false, 0.0f, false);
        double psnr = calculatePSNR(originalImg, rcasData, width, height);
        if (psnr > bestPsnr) {
            bestPsnr     = psnr;
            outUseRcas   = true;
            outSharpness = testSharpness;
        }
    }

    std::cout << "  [Auto-Tuner] Peak PSNR: " << std::fixed << std::setprecision(2)
              << bestPsnr << " dB -> ";
    if (outUseRcas) std::cout << "RCAS ON (Sharpness " << outSharpness << ")" << std::endl;
    else            std::cout << "RCAS OFF" << std::endl;

    delete[] downData;
    delete[] upData;
    delete[] rcasData;
}

// =========================================================================
// processImage
// =========================================================================
static bool processImage(
        const std::string& inFile, const std::string& outFile,
        float scale, const std::string& algo,
        bool useRcas, float sharpness, bool rcasDenoise,
        float lfga, bool tepd, int bpp,
        const std::string& paletteMatch, const std::string& paletteDither,
        const std::string& matchPaletteFrom,
        bool runPsnrOpt, float psnrDownScale, const std::string& psnrDownAlgo,
        // ICBI parameters
        int icbi_sz, int icbi_pf, bool icbi_vr, int icbi_st,
        double icbi_tm, double icbi_tc, int icbi_sc, double icbi_ts,
        double icbi_al, double icbi_bt, double icbi_gm, bool icbi_fcbi) {

    std::vector<ColorRGBA> targetPalette;
    bool hasTransparency = false;
    bool shouldMatch = false;

    if (bpp == 8) {
        if (!matchPaletteFrom.empty()) {
            if (!loadOriginalPalette(matchPaletteFrom, targetPalette, hasTransparency)) {
                std::cout << "Error: Could not load palette from " << matchPaletteFrom << std::endl;
                return false;
            }
            shouldMatch = true;
        } else {
            bool is8Bit = loadOriginalPalette(inFile, targetPalette, hasTransparency);
            if (paletteMatch == "on") {
                if (!is8Bit) {
                    std::cout << "Error: --palette-match on requested, but input is not 8-bit." << std::endl;
                    return false;
                }
                shouldMatch = true;
            } else if (paletteMatch == "auto") {
                shouldMatch = is8Bit;
            } else {
                shouldMatch = false;
            }
        }
    }

    int width, height, channels;
    unsigned char* imgData = stbi_load(inFile.c_str(), &width, &height, &channels, 4);
    if (!imgData) {
        std::cout << "Error: Could not load image: " << inFile << std::endl;
        return false;
    }

    if (runPsnrOpt && algo != "off") {
        optimizeRCAS(imgData, width, height, algo, psnrDownScale, psnrDownAlgo,
                     rcasDenoise, useRcas, sharpness,
                     icbi_sz, icbi_pf, icbi_st, icbi_tm, icbi_tc,
                     icbi_sc, icbi_ts, icbi_al, icbi_bt, icbi_gm, icbi_fcbi);
    }

    int newW = width, newH = height;
    if (algo != "off") {
        newW = std::max(1, (int)std::round(width  * scale));
        newH = std::max(1, (int)std::round(height * scale));
    }

    unsigned char* step1Data = new unsigned char[newW * newH * 4];
    float pass1_lfga = useRcas ? 0.0f : lfga;
    bool  pass1_tepd = useRcas ? false : tepd;

    if (algo == "off") {
        std::memcpy(step1Data, imgData, (size_t)newW * newH * 4);
    } else if (algo == "nearest") {
        scaleNearestNeighbor(imgData, width, height, step1Data, newW, newH);
    } else if (algo == "bilinear") {
        scaleBilinear(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
    } else if (algo == "bicubic") {
        scaleBicubic(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
    } else if (algo == "lanczos3") {
        scaleLanczos3(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
    } else if (algo == "fsr") {
        if (scale < 1.0f) {
            std::cout << "Error: FSR cannot downscale." << std::endl;
            stbi_image_free(imgData); delete[] step1Data; return false;
        }
        scaleFSR_EASU(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
    } else if (algo == "icbi") {
        scaleICBI(imgData, width, height, step1Data, newW, newH,
                  icbi_sz, icbi_pf, icbi_vr, icbi_st,
                  icbi_tm, icbi_tc, icbi_sc, icbi_ts,
                  icbi_al, icbi_bt, icbi_gm, icbi_fcbi);
    } else {
        std::cout << "Error: Unknown algorithm: " << algo << std::endl;
        stbi_image_free(imgData); delete[] step1Data; return false;
    }

    unsigned char* finalData = step1Data;
    if (useRcas) {
        finalData = new unsigned char[newW * newH * 4];
        applyFSR_RCAS(step1Data, newW, newH, finalData, sharpness, rcasDenoise, lfga, tepd);
        delete[] step1Data;
    }

    // --- SAVING ---
    if (bpp == 8) {
        if (!shouldMatch) {
            generatePalette(finalData, newW, newH, targetPalette, hasTransparency);
        }
        unsigned char* indexedData = new unsigned char[newW * newH];
        bool useFS = (paletteDither == "fs");
        quantizeAndDither(finalData, newW, newH, indexedData, targetPalette, hasTransparency, useFS);
        saveIndexedPNG(outFile.c_str(), indexedData, newW, newH, targetPalette);
        delete[] indexedData;
    } else {
        int outChannels = 4;
        unsigned char* saveBuffer = finalData;
        if (bpp == 24) {
            outChannels = 3;
            saveBuffer  = new unsigned char[(size_t)newW * newH * 3];
            for (int i = 0; i < newW * newH; ++i) {
                saveBuffer[i*3+0] = finalData[i*4+0];
                saveBuffer[i*3+1] = finalData[i*4+1];
                saveBuffer[i*3+2] = finalData[i*4+2];
            }
        }
        stbi_write_png(outFile.c_str(), newW, newH, outChannels, saveBuffer, newW * outChannels);
        if (bpp == 24) delete[] saveBuffer;
    }

    stbi_image_free(imgData);
    delete[] finalData;
    return true;
}

// =========================================================================
// Metrics mode
// =========================================================================
static bool runMetricsMode(const std::string& refFile, const std::string& distFile,
                            bool doPsnr, bool doSsim, bool doFsim) {
    int w1, h1, c1, w2, h2, c2;
    unsigned char* ref  = stbi_load(refFile.c_str(),  &w1, &h1, &c1, 4);
    unsigned char* dist = stbi_load(distFile.c_str(), &w2, &h2, &c2, 4);

    if (!ref) {
        std::cout << "Error: Could not load reference image: " << refFile << std::endl;
        if (dist) stbi_image_free(dist);
        return false;
    }
    if (!dist) {
        std::cout << "Error: Could not load distorted image: " << distFile << std::endl;
        stbi_image_free(ref);
        return false;
    }
    if (w1 != w2 || h1 != h2) {
        std::cout << "Error: Images must be the same size for metric calculation." << std::endl;
        std::cout << "  Reference : " << w1 << "x" << h1 << std::endl;
        std::cout << "  Distorted : " << w2 << "x" << h2 << std::endl;
        stbi_image_free(ref);
        stbi_image_free(dist);
        return false;
    }

    std::cout << "Computing metrics (" << w1 << "x" << h1 << ")..." << std::endl;
    MetricResults results = computeMetrics(ref, dist, w1, h1, doPsnr, doSsim, doFsim);
    printMetrics(results);

    stbi_image_free(ref);
    stbi_image_free(dist);
    return true;
}

// =========================================================================
// main
// =========================================================================
int main(int argc, char** argv) {
    std::cout << "--- Image Resizer CPU ---" << std::endl;

    std::string inputPath = "", outputPath = "", suffix = "";
    std::string algo = "fsr", rcasInput = "auto";
    std::string paletteMatch = "auto", matchPaletteFrom = "", paletteDither = "none";
    float scale = 2.0f, sharpness = 0.2f, lfga = 0.0f;
    bool rcasDenoise = false, tepd = false;
    int bpp = 32;

    // Auto-Tuner
    bool runPsnrOpt = false;
    float psnrDownScale = 0.5f;
    std::string psnrDownAlgo = "bicubic";

    // ICBI parameters (defaults match MATLAB)
    int    icbi_sz   = 8;
    int    icbi_pf   = 1;
    bool   icbi_vr   = false;
    int    icbi_st   = 20;
    double icbi_tm   = 100.0;
    double icbi_tc   = 50.0;
    int    icbi_sc   = 1;
    double icbi_ts   = 100.0;
    double icbi_al   = 1.0;
    double icbi_bt   = -1.0;
    double icbi_gm   = 5.0;
    bool   icbi_fcbi = false;

    // Metrics
    std::string metricNames = "psnr,ssim,fsim";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if      (arg == "--scale"    && i+1 < argc) scale    = std::stof(argv[++i]);
        else if (arg == "--algo"     && i+1 < argc) algo     = argv[++i];
        else if (arg == "--sharpness"&& i+1 < argc) sharpness= std::stof(argv[++i]);
        else if (arg == "--suffix"   && i+1 < argc) suffix   = argv[++i];
        else if (arg == "--rcas"     && i+1 < argc) rcasInput= argv[++i];
        else if (arg == "--rcas-denoise" && i+1 < argc) rcasDenoise = (std::string(argv[++i]) == "on");
        else if (arg == "--lfga"     && i+1 < argc) lfga     = std::stof(argv[++i]);
        else if (arg == "--tepd"     && i+1 < argc) tepd     = (std::string(argv[++i]) == "on");
        else if (arg == "--bpp"      && i+1 < argc) bpp      = std::stoi(argv[++i]);
        else if (arg == "--palette-match"       && i+1 < argc) paletteMatch     = argv[++i];
        else if (arg == "--match-palette-from"  && i+1 < argc) matchPaletteFrom = argv[++i];
        else if (arg == "--palette-dither"      && i+1 < argc) paletteDither    = argv[++i];

        // ICBI flags
        else if (arg == "--icbi-sz"  && i+1 < argc) icbi_sz  = std::stoi(argv[++i]);
        else if (arg == "--icbi-pf"  && i+1 < argc) icbi_pf  = std::stoi(argv[++i]);
        else if (arg == "--icbi-vr"  && i+1 < argc) icbi_vr  = (std::string(argv[++i]) == "on");
        else if (arg == "--icbi-st"  && i+1 < argc) icbi_st  = std::stoi(argv[++i]);
        else if (arg == "--icbi-tm"  && i+1 < argc) icbi_tm  = std::stod(argv[++i]);
        else if (arg == "--icbi-tc"  && i+1 < argc) icbi_tc  = std::stod(argv[++i]);
        else if (arg == "--icbi-sc"  && i+1 < argc) icbi_sc  = std::stoi(argv[++i]);
        else if (arg == "--icbi-ts"  && i+1 < argc) icbi_ts  = std::stod(argv[++i]);
        else if (arg == "--icbi-al"  && i+1 < argc) icbi_al  = std::stod(argv[++i]);
        else if (arg == "--icbi-bt"  && i+1 < argc) icbi_bt  = std::stod(argv[++i]);
        else if (arg == "--icbi-gm"  && i+1 < argc) icbi_gm  = std::stod(argv[++i]);
        else if (arg == "--icbi-fcbi") icbi_fcbi = true;

        // Metrics
        else if (arg == "--metric-name" && i+1 < argc) metricNames = argv[++i];

        // Auto-tuner
        else if (arg == "--rcas-max-psnr") {
            runPsnrOpt = true;
            if (i+1 < argc) {
                std::string nextArg = argv[i+1];
                if (nextArg.find("--") != 0) {
                    size_t comma = nextArg.find(',');
                    if (comma != std::string::npos) {
                        try {
                            psnrDownScale = std::stof(nextArg.substr(0, comma));
                            psnrDownAlgo  = nextArg.substr(comma + 1);
                        } catch(...) {}
                    }
                    i++;
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
                  << "  --algo fsr|icbi|lanczos3|bicubic|bilinear|nearest|off\n"
                  << "\nICBI options (only apply when --algo icbi):\n"
                  << "  --icbi-fcbi          Use FCBI only (no iterative refinement, faster)\n"
                  << "  --icbi-pf  1         Potential function: 1=curvature, 2=isophote, 3=both\n"
                  << "  --icbi-st  20        Max iterations\n"
                  << "  --icbi-tm  100       Max edge step threshold\n"
                  << "  --icbi-tc  50        Edge continuity threshold\n"
                  << "  --icbi-sc  1         Stopping: 1=change<threshold, 0=fixed iters\n"
                  << "  --icbi-ts  100       Change threshold for stopping\n"
                  << "  --icbi-al  1.0       Curvature continuity weight\n"
                  << "  --icbi-bt  -1.0      Curvature enhancement weight\n"
                  << "  --icbi-gm  5.0       Isophote smoothing weight\n"
                  << "  --icbi-vr  on|off    Verbose output\n"
                  << "\nPost-FX:\n"
                  << "  --rcas on|off  --sharpness 0.2  --rcas-denoise on|off\n"
                  << "  --lfga 0.0  --tepd on|off\n"
                  << "\nOutput:\n"
                  << "  --bpp 32|24|8  --suffix _hd\n"
                  << "\n8-bit palette:\n"
                  << "  --palette-match auto|on|off  --match-palette-from <file>\n"
                  << "  --palette-dither fs|none\n"
                  << "\nMetrics (--algo metrics):\n"
                  << "  image-resizer ref.png dist.png --algo metrics\n"
                  << "  --metric-name psnr,ssim,fsim\n"
                  << "\nAuto-Tune:\n"
                  << "  --rcas-max-psnr [0.5,bicubic]\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // Metrics mode
    // -----------------------------------------------------------------------
    if (algo == "metrics") {
        // Parse requested metrics
        bool doPsnr = false, doSsim = false, doFsim = false;
        // Tokenise metricNames by comma
        std::string tok;
        std::istringstream ss(metricNames);
        while (std::getline(ss, tok, ',')) {
            // trim whitespace
            tok.erase(0, tok.find_first_not_of(" \t"));
            tok.erase(tok.find_last_not_of(" \t") + 1);
            if (tok == "psnr") doPsnr = true;
            else if (tok == "ssim") doSsim = true;
            else if (tok == "fsim") doFsim = true;
            else std::cout << "Warning: Unknown metric '" << tok << "' ignored." << std::endl;
        }
        if (!doPsnr && !doSsim && !doFsim) {
            // default: all
            doPsnr = doSsim = doFsim = true;
        }
        return runMetricsMode(inputPath, outputPath, doPsnr, doSsim, doFsim) ? 0 : 1;
    }

    // -----------------------------------------------------------------------
    // Normal scaling mode
    // -----------------------------------------------------------------------
    bool useRcas = (rcasInput == "on" || (rcasInput == "auto" && algo == "fsr"));

    if (fs::is_directory(inputPath)) {
        if (!fs::exists(outputPath)) fs::create_directories(outputPath);
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            if (entry.is_regular_file() && entry.path().extension().string() == ".png") {
                std::string outFilePath = (fs::path(outputPath) /
                    (entry.path().stem().string() + suffix + ".png")).string();
                std::cout << " -> " << entry.path().filename().string() << std::endl;
                processImage(entry.path().string(), outFilePath,
                             scale, algo, useRcas, sharpness, rcasDenoise,
                             lfga, tepd, bpp,
                             paletteMatch, paletteDither, matchPaletteFrom,
                             runPsnrOpt, psnrDownScale, psnrDownAlgo,
                             icbi_sz, icbi_pf, icbi_vr, icbi_st,
                             icbi_tm, icbi_tc, icbi_sc, icbi_ts,
                             icbi_al, icbi_bt, icbi_gm, icbi_fcbi);
            }
        }
    } else {
        processImage(inputPath, outputPath,
                     scale, algo, useRcas, sharpness, rcasDenoise,
                     lfga, tepd, bpp,
                     paletteMatch, paletteDither, matchPaletteFrom,
                     runPsnrOpt, psnrDownScale, psnrDownAlgo,
                     icbi_sz, icbi_pf, icbi_vr, icbi_st,
                     icbi_tm, icbi_tc, icbi_sc, icbi_ts,
                     icbi_al, icbi_bt, icbi_gm, icbi_fcbi);
    }
    return 0;
}
