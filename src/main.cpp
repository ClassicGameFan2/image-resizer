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
#include "fsr_math.h"

namespace fs = std::filesystem;

// ── FSR 1.2.2 ─────────────────────────────────────────────────────────────────
extern void scaleFSR_EASU(const unsigned char* input, int inW, int inH,
                           unsigned char* output, int outW, int outH,
                           float lfga, bool tepd);
extern void applyFSR_RCAS(const unsigned char* input, int w, int h,
                           unsigned char* output, float sharpness,
                           bool useDenoise, float lfga, bool tepd);

// ── Standard scalers ──────────────────────────────────────────────────────────
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

// ── FSR 2.3.4 ─────────────────────────────────────────────────────────────────
#include "fsr2_context.h"

extern void scaleFSR2(
    const unsigned char* input, int inW, int inH,
    unsigned char* output, int outW, int outH,
    float sharpness, bool useRcas, bool rcasDenoise,
    float lfga, bool useTepd,
    float depth, bool useJitter, int numFrames,
    Fsr2JitterMode jitterMode,
    const std::set<int>& enabledPasses);

// ── FSR2 RCAS applied to uint8 data ───────────────────────────────────────────
//
// Used for cross-RCAS cases (FSR2 RCAS on non-FSR2 upscale output, or
// FSR2 RCAS as the post-step after FSR2 upscale when auto-tuner is active).
//
// Converts uint8→float, applies FSR2 RCAS in linear float space, optionally
// applies LFGA/TEPD in the same float pass (correct order: RCAS then post-fx),
// then converts back to uint8.
//
// lfga and tepd default to 0/false so this can also be used without post-fx
// (e.g. during auto-tuner PSNR sweep where LFGA/TEPD must stay off).
static void applyFSR2_RCAS_uint8(
    const unsigned char* input, int w, int h,
    unsigned char* output,
    float sharpness, bool denoise,
    float lfga = 0.0f, bool tepd = false)
{
    size_t pixels = (size_t)w * h;
    std::vector<float> floatBuf(pixels * 4);
    std::vector<float> outFloat(pixels * 4);

    for (size_t i = 0; i < pixels * 4; i++)
        floatBuf[i] = input[i] / 255.0f;

    fsr2PassRCAS(floatBuf.data(), w, h, outFloat.data(), sharpness, denoise);

    // Apply LFGA/TEPD in float space after RCAS — correct pipeline order.
    // This branch is skipped during auto-tuner PSNR sweeps (lfga=0, tepd=false).
    if (lfga > 0.0f || tepd) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                size_t idx = ((size_t)y * w + x) * 4;
                float3 color(outFloat[idx], outFloat[idx+1], outFloat[idx+2]);
                color = applyPostProcess(color, x, y, lfga, tepd);
                outFloat[idx+0] = color.x;
                outFloat[idx+1] = color.y;
                outFloat[idx+2] = color.z;
            }
        }
    }

    for (size_t i = 0; i < pixels * 4; i++)
        output[i] = (unsigned char)(clamp(outFloat[i], 0.0f, 1.0f) * 255.0f + 0.5f);
}

// ── PSNR Calculator ───────────────────────────────────────────────────────────
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
    if (mse <= 0.0000001) return 100.0;
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

// ── Auto-Tuner ────────────────────────────────────────────────────────────────
//
// Finds the RCAS sharpness level that maximizes PSNR for the given combination
// of upscaling algorithm and RCAS type.
//
// Pipeline during PSNR sweep (sterile environment):
//   1. Downscale original using downAlgo at downScale factor
//   2. Upscale back to original size using upAlgo (NO RCAS, LFGA=0, TEPD=off)
//   3. Evaluate base PSNR with no sharpening
//   4. Sweep sharpness 0.0→2.0 in 0.1 steps using rcasType
//      - sharpness=0.0 is MAXIMUM (exp2(-0.0)=1.0 multiplier)
//      - sharpness=2.0 is MINIMUM (exp2(-2.0)=0.25 multiplier)
//   5. Return the sharpness value that produced the highest PSNR
//
// rcasType:       "fsr1-rcas" or "fsr2-rcas" (already resolved, never "auto")
// rcasDenoise:    FSR1 RCAS denoising flag  — honored when rcasType="fsr1-rcas"
// fsr2RcasDenoise: FSR2 RCAS denoising flag — honored when rcasType="fsr2-rcas"
// LFGA/TEPD:      always forced to false/off during PSNR sweep for accuracy
//
// FSR2-specific params (fsr2Depth, fsr2Jitter, etc.) are used when upAlgo="fsr2"
// so the user's FSR2 settings are honored during the upscale reference step.
void optimizeRCAS(
    const unsigned char* originalImg, int width, int height,
    const std::string& upAlgo,
    float downScale, const std::string& downAlgo,
    const std::string& rcasType,
    bool rcasDenoise,
    bool fsr2RcasDenoise,
    float fsr2Depth, bool fsr2Jitter, int fsr2Frames,
    Fsr2JitterMode fsr2JitterMode,
    const std::set<int>& fsr2EnabledPasses,
    bool& outUseRcas, float& outSharpness)
{
    std::cout << "  [Auto-Tuner] Running PSNR optimization ("
              << downScale << "x " << downAlgo << " down -> "
              << upAlgo << " up -> " << rcasType << ")..." << std::endl;

    int downW = std::max(1, (int)(width  * downScale));
    int downH = std::max(1, (int)(height * downScale));

    unsigned char* downData = new unsigned char[(size_t)downW * downH * 4];
    unsigned char* upData   = new unsigned char[(size_t)width * height * 4];

    // ── Step 1: Downscale ───────────────────────────────────────────────────
    if      (downAlgo == "nearest")  scaleNearestNeighbor(originalImg, width, height, downData, downW, downH);
    else if (downAlgo == "bilinear") scaleBilinear(originalImg, width, height, downData, downW, downH, 0.0f, false);
    else if (downAlgo == "lanczos3") scaleLanczos3(originalImg, width, height, downData, downW, downH, 0.0f, false);
    else                             scaleBicubic (originalImg, width, height, downData, downW, downH, 0.0f, false);

    // ── Step 2: Upscale (NO RCAS, LFGA=0, TEPD=off) ────────────────────────
    if (upAlgo == "nearest") {
        scaleNearestNeighbor(downData, downW, downH, upData, width, height);
    } else if (upAlgo == "bilinear") {
        scaleBilinear(downData, downW, downH, upData, width, height, 0.0f, false);
    } else if (upAlgo == "bicubic") {
        scaleBicubic(downData, downW, downH, upData, width, height, 0.0f, false);
    } else if (upAlgo == "lanczos3") {
        scaleLanczos3(downData, downW, downH, upData, width, height, 0.0f, false);
    } else if (upAlgo == "fsr") {
        scaleFSR_EASU(downData, downW, downH, upData, width, height, 0.0f, false);
    } else if (upAlgo == "fsr2") {
        // Honor all FSR2 user settings for the upscale reference.
        // Strip pass 5 (RCAS) so the upscaled base is sharpening-free.
        // LFGA=0, TEPD=false for sterile PSNR environment.
        std::set<int> passesNoRcas = fsr2EnabledPasses;
        passesNoRcas.erase(FSR2_PASS_RCAS);
        scaleFSR2(downData, downW, downH, upData, width, height,
                  0.0f, false, false,   // sharpness irrelevant, RCAS off
                  0.0f, false,          // lfga=0, tepd=false
                  fsr2Depth, fsr2Jitter, fsr2Frames,
                  fsr2JitterMode, passesNoRcas);
    }

    // ── Step 3: Base PSNR (no sharpening) ──────────────────────────────────
    double bestPsnr  = calculatePSNR(originalImg, upData, width, height);
    outUseRcas       = false;
    outSharpness     = 0.0f;

    std::cout << "  [Auto-Tuner] Base PSNR (RCAS off): "
              << std::fixed << std::setprecision(2) << bestPsnr << " dB" << std::endl;

    // ── Step 4: Sweep sharpness 0.0 → 2.0 in 0.1 steps ─────────────────────
    // LFGA/TEPD are always false here — they would distort PSNR.
    // rcasDenoise / fsr2RcasDenoise are honored per the user's flag.
    unsigned char* rcasData = new unsigned char[(size_t)width * height * 4];

    for (int i = 0; i <= 20; ++i) {
        float testSharpness = (float)i / 10.0f;

        if (rcasType == "fsr2-rcas") {
            applyFSR2_RCAS_uint8(upData, width, height, rcasData,
                                  testSharpness, fsr2RcasDenoise,
                                  0.0f, false);   // lfga=0, tepd=false
        } else {
            // "fsr1-rcas"
            applyFSR_RCAS(upData, width, height, rcasData,
                           testSharpness, rcasDenoise,
                           0.0f, false);           // lfga=0, tepd=false
        }

        double psnr = calculatePSNR(originalImg, rcasData, width, height);
        if (psnr > bestPsnr) {
            bestPsnr     = psnr;
            outUseRcas   = true;
            outSharpness = testSharpness;
        }
    }

    // ── Step 5: Report result ────────────────────────────────────────────────
    std::cout << "  [Auto-Tuner] Peak PSNR: "
              << std::fixed << std::setprecision(2) << bestPsnr << " dB  ->  ";
    if (outUseRcas) {
        std::cout << rcasType << " ON, sharpness=" << outSharpness << std::endl;
    } else {
        std::cout << "RCAS OFF (base upscale is already optimal)" << std::endl;
    }

    delete[] downData;
    delete[] upData;
    delete[] rcasData;
}

// ── Parse comma-separated pass list ──────────────────────────────────────────
std::set<int> parsePassList(const std::string& s) {
    std::set<int> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { result.insert(std::stoi(token)); } catch(...) {}
    }
    return result;
}
std::set<int> allPasses() { return {0,1,2,3,4,5,6,7,8}; }

// ── Main image processor ──────────────────────────────────────────────────────
//
// resolvedPsnrRcasType: "fsr1-rcas" or "fsr2-rcas" — already resolved from
//   "auto" in main(). Used by auto-tuner to know which RCAS to test, and
//   then again in the final output step to apply the winning RCAS.
bool processImage(
    const std::string& inFile, const std::string& outFile,
    float scale, const std::string& algo,
    bool useRcas, float sharpness, bool rcasDenoise,
    float lfga, bool tepd, int bpp,
    const std::string& paletteMatch, const std::string& paletteDither,
    const std::string& matchPaletteFrom,
    bool runPsnrOpt, float psnrDownScale, const std::string& psnrDownAlgo,
    const std::string& resolvedPsnrRcasType,
    float fsr2Depth, bool fsr2Jitter, int fsr2Frames,
    Fsr2JitterMode fsr2JitterMode,
    const std::set<int>& fsr2EnabledPasses,
    float fsr2Sharpness, bool fsr2Rcas, bool fsr2RcasDenoise)
{
    // ── Palette setup (unchanged) ─────────────────────────────────────────
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
            }
        }
    }

    // ── Load image ────────────────────────────────────────────────────────
    int width, height, channels;
    unsigned char* imgData = stbi_load(inFile.c_str(), &width, &height, &channels, 4);
    if (!imgData) return false;

    // ── Output dimensions ─────────────────────────────────────────────────
    int newW = width, newH = height;
    if (algo != "off") {
        newW = std::max(1, (int)(width  * scale));
        newH = std::max(1, (int)(height * scale));
    }

    // ── Auto-Tuner ────────────────────────────────────────────────────────
    // When the auto-tuner runs it finds the PSNR-maximizing sharpness for
    // resolvedPsnrRcasType. The results are stored in atUseRcas/atSharpness
    // and used in the final output step further below.
    //
    // The auto-tuner always operates at the OUTPUT resolution (width x height)
    // by downscaling then upscaling back. LFGA/TEPD are never applied during
    // the sweep — they would add noise and distort PSNR comparisons.
    bool  atUseRcas   = false;
    float atSharpness = 0.0f;

    if (runPsnrOpt && algo != "off") {
        optimizeRCAS(imgData, width, height,
                     algo, psnrDownScale, psnrDownAlgo,
                     resolvedPsnrRcasType,
                     rcasDenoise, fsr2RcasDenoise,
                     fsr2Depth, fsr2Jitter, fsr2Frames,
                     fsr2JitterMode, fsr2EnabledPasses,
                     atUseRcas, atSharpness);
    }

    // ── Scaling and RCAS ──────────────────────────────────────────────────
    //
    // TWO PATHS:
    //
    // AUTO-TUNER PATH (runPsnrOpt == true):
    //   Scale without any RCAS or LFGA/TEPD first. Then apply the winning
    //   RCAS type with lfga/tepd so the final output matches exactly what
    //   the auto-tuner measured (minus the lfga/tepd post-fx which do not
    //   affect the PSNR-optimal sharpness decision).
    //   This path supports cross-RCAS: fsr2 upscale + fsr1-rcas, or
    //   fsr1/standard upscale + fsr2-rcas.
    //
    // NORMAL PATH (runPsnrOpt == false):
    //   Original behavior unchanged.
    //   FSR2: scaleFSR2 handles RCAS + LFGA/TEPD internally.
    //   FSR1/standard: scale then optionally applyFSR_RCAS.

    unsigned char* finalData = nullptr;

    if (runPsnrOpt) {
        // ── Auto-tuner path ───────────────────────────────────────────────

        finalData = new unsigned char[(size_t)newW * newH * 4];

        // Scale without RCAS and without LFGA/TEPD
        if (algo == "fsr2") {
            if (scale < 1.0f) {
                std::cout << "Error: FSR2 cannot downscale." << std::endl;
                stbi_image_free(imgData); delete[] finalData; return false;
            }
            // Strip pass 5 so scaleFSR2 does not apply RCAS internally
            std::set<int> passesNoRcas = fsr2EnabledPasses;
            passesNoRcas.erase(FSR2_PASS_RCAS);
            scaleFSR2(imgData, width, height, finalData, newW, newH,
                      0.0f, false, false,  // no RCAS
                      0.0f, false,          // lfga=0, tepd=false
                      fsr2Depth, fsr2Jitter, fsr2Frames,
                      fsr2JitterMode, passesNoRcas);
        } else if (algo == "off") {
            std::memcpy(finalData, imgData, (size_t)newW * newH * 4);
        } else if (algo == "nearest") {
            scaleNearestNeighbor(imgData, width, height, finalData, newW, newH);
        } else if (algo == "bilinear") {
            scaleBilinear(imgData, width, height, finalData, newW, newH, 0.0f, false);
        } else if (algo == "bicubic") {
            scaleBicubic(imgData, width, height, finalData, newW, newH, 0.0f, false);
        } else if (algo == "lanczos3") {
            scaleLanczos3(imgData, width, height, finalData, newW, newH, 0.0f, false);
        } else if (algo == "fsr") {
            if (scale < 1.0f) {
                std::cout << "Error: FSR cannot downscale." << std::endl;
                stbi_image_free(imgData); delete[] finalData; return false;
            }
            scaleFSR_EASU(imgData, width, height, finalData, newW, newH, 0.0f, false);
        }

        // Apply winning RCAS (or LFGA/TEPD alone if auto-tuner chose RCAS OFF)
        if (atUseRcas) {
            unsigned char* rcasOut = new unsigned char[(size_t)newW * newH * 4];

            if (resolvedPsnrRcasType == "fsr2-rcas") {
                // FSR2 RCAS — operates in float space via helper.
                // LFGA/TEPD applied inside helper after RCAS (correct order).
                applyFSR2_RCAS_uint8(finalData, newW, newH, rcasOut,
                                      atSharpness, fsr2RcasDenoise,
                                      lfga, tepd);
            } else {
                // FSR1 RCAS — LFGA/TEPD applied inside applyFSR_RCAS.
                // rcasDenoise honored as the user specified.
                applyFSR_RCAS(finalData, newW, newH, rcasOut,
                               atSharpness, rcasDenoise,
                               lfga, tepd);
            }

            delete[] finalData;
            finalData = rcasOut;

        } else {
            // Auto-tuner found RCAS OFF is optimal.
            // Still apply LFGA/TEPD if the user requested them.
            if (lfga > 0.0f || tepd) {
                size_t pixels = (size_t)newW * newH;
                std::vector<float> floatBuf(pixels * 4);
                for (size_t i = 0; i < pixels * 4; i++)
                    floatBuf[i] = finalData[i] / 255.0f;
                for (int y = 0; y < newH; y++) {
                    for (int x = 0; x < newW; x++) {
                        size_t idx = ((size_t)y * newW + x) * 4;
                        float3 color(floatBuf[idx], floatBuf[idx+1], floatBuf[idx+2]);
                        color = applyPostProcess(color, x, y, lfga, tepd);
                        floatBuf[idx+0] = color.x;
                        floatBuf[idx+1] = color.y;
                        floatBuf[idx+2] = color.z;
                    }
                }
                unsigned char* ppOut = new unsigned char[pixels * 4];
                for (size_t i = 0; i < pixels * 4; i++)
                    ppOut[i] = (unsigned char)(clamp(floatBuf[i], 0.0f, 1.0f) * 255.0f + 0.5f);
                delete[] finalData;
                finalData = ppOut;
            }
        }

    } else {
        // ── Normal path (no auto-tuner): original behavior unchanged ──────

        if (algo == "fsr2") {
            if (scale < 1.0f) {
                std::cout << "Error: FSR2 cannot downscale. Use scale >= 1.0." << std::endl;
                stbi_image_free(imgData); return false;
            }
            finalData = new unsigned char[(size_t)newW * newH * 4];
            // scaleFSR2 handles RCAS (pass 5) + LFGA/TEPD internally
            scaleFSR2(imgData, width, height, finalData, newW, newH,
                      fsr2Sharpness, fsr2Rcas, fsr2RcasDenoise,
                      lfga, tepd,
                      fsr2Depth, fsr2Jitter, fsr2Frames,
                      fsr2JitterMode, fsr2EnabledPasses);
        } else {
            // FSR1/standard path
            unsigned char* step1Data = new unsigned char[(size_t)newW * newH * 4];
            // If RCAS will run, hold LFGA/TEPD back so they apply in the RCAS step
            float pass1_lfga = useRcas ? 0.0f : lfga;
            bool  pass1_tepd = useRcas ? false : tepd;

            if      (algo == "off")      std::memcpy(step1Data, imgData, (size_t)newW * newH * 4);
            else if (algo == "nearest")  scaleNearestNeighbor(imgData, width, height, step1Data, newW, newH);
            else if (algo == "bilinear") scaleBilinear(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
            else if (algo == "bicubic")  scaleBicubic (imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
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
                unsigned char* rcasOut = new unsigned char[(size_t)newW * newH * 4];
                applyFSR_RCAS(step1Data, newW, newH, rcasOut,
                               sharpness, rcasDenoise, lfga, tepd);
                delete[] step1Data;
                finalData = rcasOut;
            }
        }
    }

    // ── Save ─────────────────────────────────────────────────────────────
    if (bpp == 8) {
        if (!shouldMatch)
            generatePalette(finalData, newW, newH, targetPalette, hasTransparency);
        unsigned char* indexedData = new unsigned char[(size_t)newW * newH];
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
            saveBuffer = new unsigned char[(size_t)newW * newH * 3];
            for (int i = 0; i < newW * newH; i++) {
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

// ── Entry point ───────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::cout << "--- Image Resizer CPU ---" << std::endl;

    // ── FSR1 / general options ─────────────────────────────────────────────
    std::string inputPath = "", outputPath = "", suffix = "";
    std::string algo = "fsr", rcasInput = "auto";
    std::string paletteMatch = "auto", matchPaletteFrom = "", paletteDither = "none";
    float scale = 2.0f, sharpness = 0.2f, lfga = 0.0f;
    bool rcasDenoise = false, tepd = false;
    int bpp = 32;

    // ── Auto-Tuner options ─────────────────────────────────────────────────
    bool runPsnrOpt = false;
    float psnrDownScale = 0.5f;
    std::string psnrDownAlgo = "bicubic";
    // "auto" is resolved after all args are parsed:
    //   fsr2 algo  → "fsr2-rcas"
    //   other algo → "fsr1-rcas"
    std::string psnrRcasType = "auto";

    // ── FSR2 options ───────────────────────────────────────────────────────
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

        // ── General ───────────────────────────────────────────────────────
        if      (arg == "--scale"     && i+1<argc) scale     = std::stof(argv[++i]);
        else if (arg == "--algo"      && i+1<argc) algo      = argv[++i];
        else if (arg == "--sharpness" && i+1<argc) sharpness = std::stof(argv[++i]);
        else if (arg == "--suffix"    && i+1<argc) suffix    = argv[++i];
        else if (arg == "--rcas"      && i+1<argc) rcasInput = argv[++i];
        else if (arg == "--rcas-denoise"   && i+1<argc) rcasDenoise = (std::string(argv[++i]) == "on");
        else if (arg == "--lfga"      && i+1<argc) lfga      = std::stof(argv[++i]);
        else if (arg == "--tepd"      && i+1<argc) tepd      = (std::string(argv[++i]) == "on");
        else if (arg == "--bpp"       && i+1<argc) bpp       = std::stoi(argv[++i]);
        else if (arg == "--palette-match"       && i+1<argc) paletteMatch    = argv[++i];
        else if (arg == "--match-palette-from"  && i+1<argc) matchPaletteFrom = argv[++i];
        else if (arg == "--palette-dither"      && i+1<argc) paletteDither   = argv[++i];

        // ── Auto-Tuner ────────────────────────────────────────────────────
        // Syntax: --rcas-max-psnr [value]
        // value can be any of:
        //   fsr1-rcas                    → just set RCAS type
        //   fsr2-rcas                    → just set RCAS type
        //   0.5,bicubic                  → set downscale factor and algo
        //   0.5,bicubic,fsr1-rcas        → set all three
        //   0.5,bicubic,fsr2-rcas        → set all three
        else if (arg == "--rcas-max-psnr") {
            runPsnrOpt = true;
            if (i + 1 < argc) {
                std::string nextArg = argv[i+1];
                if (nextArg.find("--") != 0) {
                    if (nextArg == "fsr1-rcas" || nextArg == "fsr2-rcas") {
                        // Just a RCAS type, no scale/algo override
                        psnrRcasType = nextArg;
                    } else {
                        // downscale,algo[,rcastype]
                        size_t comma1 = nextArg.find(',');
                        if (comma1 != std::string::npos) {
                            try {
                                psnrDownScale = std::stof(nextArg.substr(0, comma1));
                                std::string rest = nextArg.substr(comma1 + 1);
                                size_t comma2 = rest.find(',');
                                if (comma2 != std::string::npos) {
                                    psnrDownAlgo = rest.substr(0, comma2);
                                    std::string rcasTok = rest.substr(comma2 + 1);
                                    if (rcasTok == "fsr1-rcas" || rcasTok == "fsr2-rcas")
                                        psnrRcasType = rcasTok;
                                } else {
                                    psnrDownAlgo = rest;
                                }
                            } catch(...) {}
                        }
                    }
                    i++; // consume the argument
                }
            }
        }

        // ── FSR2 options ──────────────────────────────────────────────────
        else if (arg == "--fsr2-depth"        && i+1<argc) fsr2Depth        = std::stof(argv[++i]);
        else if (arg == "--fsr2-jitter"       && i+1<argc) fsr2Jitter       = (std::string(argv[++i]) == "on");
        else if (arg == "--fsr2-frames"       && i+1<argc) fsr2Frames       = std::stoi(argv[++i]);
        else if (arg == "--fsr2-jitter-mode"  && i+1<argc) {
            std::string m = argv[++i];
            if      (m == "bilinear")  fsr2JitterMode = Fsr2JitterMode::BILINEAR;
            else if (m == "bicubic")   fsr2JitterMode = Fsr2JitterMode::BICUBIC;
            else if (m == "lanczos3")  fsr2JitterMode = Fsr2JitterMode::LANCZOS3;
        }
        else if (arg == "--fsr2-sharpness"    && i+1<argc) fsr2Sharpness    = std::stof(argv[++i]);
        else if (arg == "--fsr2-rcas"         && i+1<argc) fsr2Rcas         = (std::string(argv[++i]) == "on");
        else if (arg == "--fsr2-rcas-denoise" && i+1<argc) fsr2RcasDenoise  = (std::string(argv[++i]) == "on");
        else if (arg == "--onlyenablepasses"  && i+1<argc) fsr2EnabledPasses = parsePassList(argv[++i]);

        else if (inputPath.empty())  inputPath  = arg;
        else if (outputPath.empty()) outputPath = arg;
    }

    if (inputPath.empty() || outputPath.empty()) {
        std::cout << "Usage: image-resizer <input> <output> [options]\n\n";
        std::cout << "=== FSR 1.2.2 / Standard Scalers ===\n";
        std::cout << "  --scale 2.0\n";
        std::cout << "  --algo fsr|lanczos3|bicubic|bilinear|nearest|off\n";
        std::cout << "  --rcas on|off  --sharpness 0.2  --rcas-denoise on|off\n";
        std::cout << "  --lfga 0.0  --tepd on|off\n";
        std::cout << "  --bpp 32|24|8  --suffix _hd\n";
        std::cout << "  --palette-match auto|on|off  --match-palette-from <file>\n";
        std::cout << "  --palette-dither fs|none\n\n";
        std::cout << "=== FSR 2.3.4 (--algo fsr2) ===\n";
        std::cout << "  --fsr2-jitter on|off            (default on)\n";
        std::cout << "  --fsr2-frames 32                (default 32)\n";
        std::cout << "  --fsr2-jitter-mode bilinear|bicubic|lanczos3  (default bicubic)\n";
        std::cout << "  --fsr2-depth 0.5                (default 0.5)\n";
        std::cout << "  --fsr2-sharpness 0.2            (default 0.2, 0.0=max, 2.0=min)\n";
        std::cout << "  --fsr2-rcas on|off              (default on)\n";
        std::cout << "  --fsr2-rcas-denoise on|off      (default off)\n";
        std::cout << "  --onlyenablepasses 0,1,2,3,4,5,6,7,8  (all by default)\n";
        std::cout << "    Passes: 0=DepthClip 1=ReconstructDepth 2=Lock\n";
        std::cout << "            3=Accumulate 4=AccumulateSharpen 5=RCAS\n";
        std::cout << "            6=LuminancePyramid 7=GenerateReactive 8=TCR\n\n";
        std::cout << "=== Auto-Tuner ===\n";
        std::cout << "  --rcas-max-psnr [arg]\n";
        std::cout << "    arg is optional and can be:\n";
        std::cout << "      fsr1-rcas                   use FSR1 RCAS for sweep\n";
        std::cout << "      fsr2-rcas                   use FSR2 RCAS for sweep\n";
        std::cout << "      0.5,bicubic                 downscale factor + algo\n";
        std::cout << "      0.5,bicubic,fsr1-rcas       all three options\n";
        std::cout << "      0.5,bicubic,fsr2-rcas       all three options\n";
        std::cout << "    Default RCAS type: fsr1-rcas for non-FSR2 algos,\n";
        std::cout << "                       fsr2-rcas for --algo fsr2\n";
        std::cout << "    LFGA/TEPD are always disabled during PSNR sweep.\n";
        std::cout << "    rcas-denoise / fsr2-rcas-denoise flags are honored.\n";
        return 1;
    }

    // ── Post-parse setup ───────────────────────────────────────────────────

    // FSR1 RCAS on/off
    bool useRcas = (rcasInput == "on" || (rcasInput == "auto" && algo == "fsr"));

    // FSR2 RCAS on/off: controlled exclusively via enabledPasses (pass 5).
    // sharpness=0.0 means maximum sharpening, not off.
    if (!fsr2Rcas) {
        fsr2EnabledPasses.erase(FSR2_PASS_RCAS);
    }
    if (!fsr2EnabledPasses.count(FSR2_PASS_RCAS)) {
        fsr2Rcas = false;
    }

    // Resolve "auto" psnrRcasType based on the upscaling algorithm.
    // This must happen after --algo is fully parsed.
    std::string resolvedPsnrRcasType = psnrRcasType;
    if (resolvedPsnrRcasType == "auto") {
        resolvedPsnrRcasType = (algo == "fsr2") ? "fsr2-rcas" : "fsr1-rcas";
    }

    // ── Dispatch ──────────────────────────────────────────────────────────
    auto processOne = [&](const std::string& inPath, const std::string& outPath) {
        return processImage(
            inPath, outPath, scale, algo,
            useRcas, sharpness, rcasDenoise,
            lfga, tepd, bpp,
            paletteMatch, paletteDither, matchPaletteFrom,
            runPsnrOpt, psnrDownScale, psnrDownAlgo,
            resolvedPsnrRcasType,
            fsr2Depth, fsr2Jitter, fsr2Frames,
            fsr2JitterMode, fsr2EnabledPasses,
            fsr2Sharpness, fsr2Rcas, fsr2RcasDenoise);
    };

    if (fs::is_directory(inputPath)) {
        if (!fs::exists(outputPath)) fs::create_directories(outputPath);
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            if (entry.is_regular_file() &&
                entry.path().extension().string() == ".png") {
                std::string outFilePath = (fs::path(outputPath) /
                    (entry.path().stem().string() + suffix + ".png")).string();
                std::cout << " -> " << entry.path().filename().string() << std::endl;
                processOne(entry.path().string(), outFilePath);
            }
        }
    } else {
        processOne(inputPath, outputPath);
    }
    return 0;
}
