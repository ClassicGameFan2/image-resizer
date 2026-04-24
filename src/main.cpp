#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iomanip>
#include "stb_image.h"
#include "stb_image_write.h"
#include "dither.h"

namespace fs = std::filesystem;

// ── FSR 1.2.2 ─────────────────────────────────────────────────────────────────
extern void scaleFSR_EASU(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH, float lfga, bool tepd);
extern void applyFSR_RCAS(const unsigned char* input, int w, int h, unsigned char* output, float sharpness, bool useDenoise, float lfga, bool tepd);

// ── Standard scalers ──────────────────────────────────────────────────────────
extern void scaleNearestNeighbor(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH);
extern void scaleBilinear(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH, float lfga, bool tepd);
extern void scaleBicubic(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH, float lfga, bool tepd);
extern void scaleLanczos3(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH, float lfga, bool tepd);

// ── FSR 2.3.4 ─────────────────────────────────────────────────────────────────
#include "fsr2_context.h"      // includes fsr2_jitter_resample.h etc.

void scaleFSR2(
    const unsigned char* input, int inW, int inH,
    unsigned char* output, int outW, int outH,
    float sharpness, bool useRcas, bool rcasDenoise,
    float lfga, bool useTepd,
    float depth, bool useJitter, int numFrames,
    Fsr2JitterMode jitterMode,
    const std::set<int>& enabledPasses);

// ── PSNR Calculator ───────────────────────────────────────────────────────────
double calculatePSNR(const unsigned char* img1, const unsigned char* img2, int width, int height) {
    double mse = 0.0;
    for (int i = 0; i < width * height * 4; i += 4) {
        double dr = (double)img1[i+0] - (double)img2[i+0];
        double dg = (double)img1[i+1] - (double)img2[i+1];
        double db = (double)img1[i+2] - (double)img2[i+2];
        mse += dr*dr + dg*dg + db*db;
    }
    mse /= (width * height * 3.0);
    if (mse <= 0.0000001) return 100.0;
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

// ── Auto-Tuner ────────────────────────────────────────────────────────────────
void optimizeRCAS(const unsigned char* originalImg, int width, int height, const std::string& upAlgo,
                  float downScale, const std::string& downAlgo, bool rcasDenoise,
                  bool& outUseRcas, float& outSharpness) {
    std::cout << "  [Auto-Tuner] Running PSNR optimization pass (" << downScale << "x " << downAlgo << " downscale -> " << upAlgo << " upscale)..." << std::endl;

    int downW = std::max(1, (int)(width * downScale));
    int downH = std::max(1, (int)(height * downScale));

    unsigned char* downData = new unsigned char[downW * downH * 4];
    unsigned char* upData = new unsigned char[width * height * 4];

    if (downAlgo == "nearest") scaleNearestNeighbor(originalImg, width, height, downData, downW, downH);
    else if (downAlgo == "bilinear") scaleBilinear(originalImg, width, height, downData, downW, downH, 0.0f, false);
    else if (downAlgo == "lanczos3") scaleLanczos3(originalImg, width, height, downData, downW, downH, 0.0f, false);
    else scaleBicubic(originalImg, width, height, downData, downW, downH, 0.0f, false);

    if (upAlgo == "nearest") scaleNearestNeighbor(downData, downW, downH, upData, width, height);
    else if (upAlgo == "bilinear") scaleBilinear(downData, downW, downH, upData, width, height, 0.0f, false);
    else if (upAlgo == "bicubic") scaleBicubic(downData, downW, downH, upData, width, height, 0.0f, false);
    else if (upAlgo == "lanczos3") scaleLanczos3(downData, downW, downH, upData, width, height, 0.0f, false);
    else if (upAlgo == "fsr") scaleFSR_EASU(downData, downW, downH, upData, width, height, 0.0f, false);

    double bestPsnr = calculatePSNR(originalImg, upData, width, height);
    outUseRcas = false;
    outSharpness = 0.0f;

    unsigned char* rcasData = new unsigned char[width * height * 4];
    for (int i = 0; i <= 20; ++i) {
        float testSharpness = i / 10.0f;
        applyFSR_RCAS(upData, width, height, rcasData, testSharpness, false, 0.0f, false);
        double psnr = calculatePSNR(originalImg, rcasData, width, height);
        if (psnr > bestPsnr) {
            bestPsnr = psnr;
            outUseRcas = true;
            outSharpness = testSharpness;
        }
    }

    std::cout << "  [Auto-Tuner] Peak PSNR: " << std::fixed << std::setprecision(2) << bestPsnr << " dB -> ";
    if (outUseRcas) std::cout << "Winning Setting: RCAS ON (Sharpness " << outSharpness << ")" << std::endl;
    else std::cout << "Winning Setting: RCAS OFF" << std::endl;

    delete[] downData;
    delete[] upData;
    delete[] rcasData;
}

// ── Parse pass list from string "0,1,2,3,4,5,6,7,8" ──────────────────────────
std::set<int> parsePassList(const std::string& s) {
    std::set<int> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { result.insert(std::stoi(token)); } catch(...) {}
    }
    return result;
}

std::set<int> allPasses() {
    return {0,1,2,3,4,5,6,7,8};
}

// ── Main image processor ──────────────────────────────────────────────────────
bool processImage(
    const std::string& inFile, const std::string& outFile,
    float scale, const std::string& algo,
    bool useRcas, float sharpness, bool rcasDenoise,
    float lfga, bool tepd, int bpp,
    const std::string& paletteMatch, const std::string& paletteDither,
    const std::string& matchPaletteFrom,
    bool runPsnrOpt, float psnrDownScale, const std::string& psnrDownAlgo,
    // FSR2 specific
    float fsr2Depth, bool fsr2Jitter, int fsr2Frames,
    Fsr2JitterMode fsr2JitterMode,
    const std::set<int>& fsr2EnabledPasses,
    float fsr2Sharpness, bool fsr2Rcas, bool fsr2RcasDenoise)
{
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
    if (!imgData) return false;

    // Auto-Tuner (FSR1 only)
    if (runPsnrOpt && algo != "off" && algo != "fsr2") {
        optimizeRCAS(imgData, width, height, algo, psnrDownScale, psnrDownAlgo, rcasDenoise, useRcas, sharpness);
    }

    // ── Compute output size ──
    int newW = width, newH = height;
    if (algo != "off") {
        newW = std::max(1, (int)(width * scale));
        newH = std::max(1, (int)(height * scale));
    }

    unsigned char* finalData = nullptr;

    // ── FSR 2.3.4 path ──
    if (algo == "fsr2") {
        if (scale < 1.0f) {
            std::cout << "Error: FSR2 cannot downscale. Use scale >= 1.0." << std::endl;
            stbi_image_free(imgData);
            return false;
        }
        finalData = new unsigned char[newW * newH * 4];
        scaleFSR2(
            imgData, width, height,
            finalData, newW, newH,
            fsr2Sharpness, fsr2Rcas, fsr2RcasDenoise,
            lfga, tepd,
            fsr2Depth, fsr2Jitter, fsr2Frames,
            fsr2JitterMode, fsr2EnabledPasses);
    }
    // ── FSR 1.2.2 / standard scaler path ──
    else {
        unsigned char* step1Data = new unsigned char[newW * newH * 4];
        float pass1_lfga = useRcas ? 0.0f : lfga;
        bool pass1_tepd = useRcas ? false : tepd;

        if (algo == "off") std::memcpy(step1Data, imgData, (size_t)newW * newH * 4);
        else if (algo == "nearest") scaleNearestNeighbor(imgData, width, height, step1Data, newW, newH);
        else if (algo == "bilinear") scaleBilinear(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
        else if (algo == "bicubic") scaleBicubic(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
        else if (algo == "lanczos3") scaleLanczos3(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
        else if (algo == "fsr") {
            if (scale < 1.0f) {
                std::cout << "Error: FSR mathematically cannot downscale." << std::endl;
                stbi_image_free(imgData); delete[] step1Data; return false;
            }
            scaleFSR_EASU(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
        }

        finalData = step1Data;
        if (useRcas) {
            unsigned char* rcasOut = new unsigned char[newW * newH * 4];
            applyFSR_RCAS(step1Data, newW, newH, rcasOut, sharpness, rcasDenoise, lfga, tepd);
            delete[] step1Data;
            finalData = rcasOut;
        }
    }

    // ── Save ──
    if (bpp == 8) {
        if (!shouldMatch) generatePalette(finalData, newW, newH, targetPalette, hasTransparency);
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
            saveBuffer = new unsigned char[newW * newH * 3];
            for (int i = 0; i < newW * newH; i++) {
                saveBuffer[i * 3 + 0] = finalData[i * 4 + 0];
                saveBuffer[i * 3 + 1] = finalData[i * 4 + 1];
                saveBuffer[i * 3 + 2] = finalData[i * 4 + 2];
            }
        }
        stbi_write_png(outFile.c_str(), newW, newH, outChannels, saveBuffer, newW * outChannels);
        if (bpp == 24) delete[] saveBuffer;
    }

    stbi_image_free(imgData);
    delete[] finalData;
    return true;
}

int main(int argc, char** argv) {
    std::cout << "--- Image Resizer CPU ---" << std::endl;

    // ── FSR 1 / general options ───────────────────────────────────────────────
    std::string inputPath = "", outputPath = "", suffix = "", algo = "fsr", rcasInput = "auto";
    std::string paletteMatch = "auto", matchPaletteFrom = "", paletteDither = "none";
    float scale = 2.0f, sharpness = 0.2f, lfga = 0.0f;
    bool rcasDenoise = false, tepd = false;
    int bpp = 32;
    bool runPsnrOpt = false;
    float psnrDownScale = 0.5f;
    std::string psnrDownAlgo = "bicubic";

    // ── FSR 2.3.4 options ─────────────────────────────────────────────────────
    float fsr2Depth = 0.5f;
    bool fsr2Jitter = true;
    int fsr2Frames = 32;
    Fsr2JitterMode fsr2JitterMode = Fsr2JitterMode::BICUBIC;
    std::set<int> fsr2EnabledPasses = allPasses();
    float fsr2Sharpness = 0.2f;
    bool fsr2Rcas = true;
    bool fsr2RcasDenoise = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        // ── General ──
        if (arg == "--scale" && i + 1 < argc)         scale = std::stof(argv[++i]);
        else if (arg == "--algo" && i + 1 < argc)     algo = argv[++i];
        else if (arg == "--sharpness" && i + 1 < argc) sharpness = std::stof(argv[++i]);
        else if (arg == "--suffix" && i + 1 < argc)   suffix = argv[++i];
        else if (arg == "--rcas" && i + 1 < argc)     rcasInput = argv[++i];
        else if (arg == "--rcas-denoise" && i + 1 < argc) rcasDenoise = (std::string(argv[++i]) == "on");
        else if (arg == "--lfga" && i + 1 < argc)     lfga = std::stof(argv[++i]);
        else if (arg == "--tepd" && i + 1 < argc)     tepd = (std::string(argv[++i]) == "on");
        else if (arg == "--bpp" && i + 1 < argc)      bpp = std::stoi(argv[++i]);
        else if (arg == "--palette-match" && i + 1 < argc)    paletteMatch = argv[++i];
        else if (arg == "--match-palette-from" && i + 1 < argc) matchPaletteFrom = argv[++i];
        else if (arg == "--palette-dither" && i + 1 < argc)   paletteDither = argv[++i];
        else if (arg == "--rcas-max-psnr") {
            runPsnrOpt = true;
            if (i + 1 < argc) {
                std::string nextArg = argv[i+1];
                if (nextArg.find("--") != 0) {
                    size_t comma = nextArg.find(',');
                    if (comma != std::string::npos) {
                        try {
                            psnrDownScale = std::stof(nextArg.substr(0, comma));
                            psnrDownAlgo = nextArg.substr(comma + 1);
                        } catch(...) {}
                    }
                    i++;
                }
            }
        }

        // ── FSR 2.3.4 specific ──
        else if (arg == "--fsr2-depth" && i + 1 < argc)
            fsr2Depth = std::stof(argv[++i]);
        else if (arg == "--fsr2-jitter" && i + 1 < argc)
            fsr2Jitter = (std::string(argv[++i]) == "on");
        else if (arg == "--fsr2-frames" && i + 1 < argc)
            fsr2Frames = std::stoi(argv[++i]);
        else if (arg == "--fsr2-jitter-mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "bilinear")      fsr2JitterMode = Fsr2JitterMode::BILINEAR;
            else if (m == "bicubic")  fsr2JitterMode = Fsr2JitterMode::BICUBIC;
            else if (m == "lanczos3") fsr2JitterMode = Fsr2JitterMode::LANCZOS3;
        }
        else if (arg == "--fsr2-sharpness" && i + 1 < argc)
            fsr2Sharpness = std::stof(argv[++i]);
        else if (arg == "--fsr2-rcas" && i + 1 < argc)
            fsr2Rcas = (std::string(argv[++i]) == "on");
        else if (arg == "--fsr2-rcas-denoise" && i + 1 < argc)
            fsr2RcasDenoise = (std::string(argv[++i]) == "on");

        // ── Pass selector: --onlyenablepasses 0,1,2,3,4,5 ──
        else if (arg == "--onlyenablepasses" && i + 1 < argc) {
            fsr2EnabledPasses = parsePassList(argv[++i]);
        }

        else if (inputPath.empty())  inputPath = arg;
        else if (outputPath.empty()) outputPath = arg;
    }

    if (inputPath.empty() || outputPath.empty()) {
        std::cout << "Usage: image-resizer <in> <out> [options]" << std::endl;
        std::cout << std::endl;
        std::cout << "=== FSR 1.2.2 / Standard Scalers ===" << std::endl;
        std::cout << "Scale:    --scale 2.0 --algo fsr|lanczos3|bicubic|bilinear|nearest|off" << std::endl;
        std::cout << "PostFx:   --rcas on|off --sharpness 0.2 --rcas-denoise on|off" << std::endl;
        std::cout << "          --lfga 0.0 --tepd on|off" << std::endl;
        std::cout << "Output:   --bpp 32|24|8 --suffix _hd" << std::endl;
        std::cout << "8-Bit:    --palette-match auto|on|off --match-palette-from <file>" << std::endl;
        std::cout << "          --palette-dither fs|none" << std::endl;
        std::cout << "Tune:     --rcas-max-psnr [0.5,bicubic]" << std::endl;
        std::cout << std::endl;
        std::cout << "=== FSR 2.3.4 (--algo fsr2) ===" << std::endl;
        std::cout << "          --scale 2.0            Output scale factor (default 2.0)" << std::endl;
        std::cout << "          --fsr2-jitter on|off   Enable Halton jitter frames (default on)" << std::endl;
        std::cout << "          --fsr2-frames 32       Number of jitter frames (default 32)" << std::endl;
        std::cout << "          --fsr2-jitter-mode bilinear|bicubic|lanczos3 (default bicubic)" << std::endl;
        std::cout << "          --fsr2-depth 0.5       Scene depth [0..1] for flat scenes (default 0.5)" << std::endl;
        std::cout << "          --fsr2-sharpness 0.2   RCAS sharpness in stops (default 0.2)" << std::endl;
        std::cout << "          --fsr2-rcas on|off     Enable FSR2 RCAS pass (default on)" << std::endl;
        std::cout << "          --fsr2-rcas-denoise on|off  RCAS denoising (default off)" << std::endl;
        std::cout << "          --onlyenablepasses 0,1,2,3,4,5,6,7,8  (all by default)" << std::endl;
        std::cout << "                Passes: 0=DepthClip 1=ReconstructDepth 2=Lock" << std::endl;
        std::cout << "                        3=Accumulate 4=AccumulateSharpen 5=RCAS" << std::endl;
        std::cout << "                        6=LuminancePyramid 7=GenerateReactive 8=TCR" << std::endl;
        std::cout << "                Example: --onlyenablepasses 3,5  (Accumulate + RCAS only)" << std::endl;
        return 1;
    }

    bool useRcas = (rcasInput == "on" || (rcasInput == "auto" && algo == "fsr"));

    // For FSR2 algo, ensure RCAS is controlled by fsr2-rcas flag, not the FSR1 rcas flag
    if (algo == "fsr2") {
        // Apply --rcas-max-psnr sharpness to fsr2 if user passed it
        // (fsr2Sharpness is the dedicated flag, but we respect --sharpness too if set)
    }

    // RCAS (pass 5) on/off is controlled exclusively via fsr2EnabledPasses.
    // --fsr2-rcas off removes pass 5 from the set so fsr2Dispatch never runs it.
    // --fsr2-rcas on keeps pass 5 in the set (default).
    // This separation means sharpness=0.0 correctly means MAXIMUM sharpening,
    // not "off". The sharpness value is only meaningful when pass 5 is enabled.
    if (!fsr2Rcas) {
        fsr2EnabledPasses.erase(FSR2_PASS_RCAS);
    }
    // If user specified --onlyenablepasses without pass 5, respect that too
    if (!fsr2EnabledPasses.count(FSR2_PASS_RCAS)) {
        fsr2Rcas = false;
    }

    auto processOne = [&](const std::string& inPath, const std::string& outPath) {
        return processImage(
            inPath, outPath, scale, algo,
            useRcas, sharpness, rcasDenoise,
            lfga, tepd, bpp,
            paletteMatch, paletteDither, matchPaletteFrom,
            runPsnrOpt, psnrDownScale, psnrDownAlgo,
            fsr2Depth, fsr2Jitter, fsr2Frames,
            fsr2JitterMode, fsr2EnabledPasses,
            fsr2Sharpness, fsr2Rcas, fsr2RcasDenoise);
    };

    if (fs::is_directory(inputPath)) {
        if (!fs::exists(outputPath)) fs::create_directories(outputPath);
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            if (entry.is_regular_file() && entry.path().extension().string() == ".png") {
                std::string outFilePath = (fs::path(outputPath) / (entry.path().stem().string() + suffix + ".png")).string();
                std::cout << " -> " << entry.path().filename().string() << std::endl;
                processOne(entry.path().string(), outFilePath);
            }
        }
    } else {
        processOne(inputPath, outputPath);
    }
    return 0;
}
