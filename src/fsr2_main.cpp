// =============================================================================
// fsr2_main.cpp
// FSR 2.3.4 CPU Port — Static image upscaling entry point
//
// This is the top-level function called from main.cpp when --algo fsr2 is used.
// It handles:
//   1. Downscaling the input image to render resolution (the "render" step)
//   2. Generating synthetic jitter frames using the Halton(2,3) sequence
//   3. Running all FSR2 passes for each jitter frame
//   4. Returning the final upscaled output
// =============================================================================
#include "fsr2_context.h"
#include "fsr2_jitter.h"
#include "fsr2_jitter_resample.h"
#include "fsr_math.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <set>
#include <iostream>
#include <algorithm>

// ── uint8 ↔ float conversion helpers ────────────────────────────────────────
static void uint8ToFloat(const unsigned char* src, float* dst, int count) {
    for (int i = 0; i < count * 4; i++)
        dst[i] = src[i] / 255.0f;
}
static void floatToUint8(const float* src, unsigned char* dst, int count) {
    for (int i = 0; i < count * 4; i++)
        dst[i] = (unsigned char)(clamp(src[i], 0.0f, 1.0f) * 255.0f + 0.5f);
}

// ── Bicubic downscale (float RGBA) ───────────────────────────────────────────
// Uses the same Catmull-Rom kernel as scalers.cpp
static float cubicWF(float t) {
    t = std::abs(t);
    if (t < 1.0f) return 1.5f*t*t*t - 2.5f*t*t + 1.0f;
    if (t < 2.0f) return -0.5f*t*t*t + 2.5f*t*t - 4.0f*t + 2.0f;
    return 0.0f;
}

static void bicubicDownscaleF(const float* src, int srcW, int srcH,
                               float* dst, int dstW, int dstH) {
    float scaleX = (float)srcW / (float)dstW;
    float scaleY = (float)srcH / (float)dstH;
    float filterScaleX = std::min(1.0f, (float)dstW / (float)srcW);
    float filterScaleY = std::min(1.0f, (float)dstH / (float)srcH);
    float radiusX = 2.0f / filterScaleX;
    float radiusY = 2.0f / filterScaleY;

    for (int dy = 0; dy < dstH; dy++) {
        float srcY = ((float)dy + 0.5f) / scaleY - 0.5f;
        int yMin = std::max(0, (int)std::floor(srcY - radiusY + 1));
        int yMax = std::min(srcH - 1, (int)std::floor(srcY + radiusY));

        for (int dx = 0; dx < dstW; dx++) {
            float srcX = ((float)dx + 0.5f) / scaleX - 0.5f;
            int xMin = std::max(0, (int)std::floor(srcX - radiusX + 1));
            int xMax = std::min(srcW - 1, (int)std::floor(srcX + radiusX));

            double r=0,g=0,b=0,a=0,wSum=0;
            for (int cy = yMin; cy <= yMax; cy++) {
                float wy = cubicWF((srcY - cy) * filterScaleY);
                for (int cx = xMin; cx <= xMax; cx++) {
                    float w = cubicWF((srcX - cx) * filterScaleX) * wy;
                    size_t sIdx = ((size_t)cy * srcW + cx) * 4;
                    r += src[sIdx+0] * w;
                    g += src[sIdx+1] * w;
                    b += src[sIdx+2] * w;
                    a += src[sIdx+3] * w;
                    wSum += w;
                }
            }
            if (wSum < 1e-6f) wSum = 1.0f;
            size_t dIdx = ((size_t)dy * dstW + dx) * 4;
            dst[dIdx+0] = clamp((float)(r/wSum), 0.0f, 1.0f);
            dst[dIdx+1] = clamp((float)(g/wSum), 0.0f, 1.0f);
            dst[dIdx+2] = clamp((float)(b/wSum), 0.0f, 1.0f);
            dst[dIdx+3] = clamp((float)(a/wSum), 0.0f, 1.0f);
        }
    }
}

// ================================================================
// scaleFSR2: Main entry point for FSR2 upscaling of a static image.
//
// Parameters:
//   input        — source image (uint8 RGBA, inW x inH)
//   inW, inH     — source dimensions
//   output       — destination (uint8 RGBA, outW x outH)
//   outW, outH   — output dimensions
//   sharpness    — RCAS sharpness in stops (0 = off)
//   useRcas      — enable RCAS pass
//   rcasDenoise  — enable RCAS denoise
//   lfga         — LFGA amount (passed to applyPostProcess)
//   useTepd      — TEPD dither (passed to applyPostProcess)
//   depth        — flat depth [0..1] for the virtual scene (default 0.5)
//   useJitter    — enable Halton jitter frame generation
//   numFrames    — number of jitter frames to accumulate
//   jitterMode   — bilinear/bicubic/lanczos3 for sub-pixel shift
//   enabledPasses — which FSR2 passes to run (all 9 by default)
// ================================================================
void scaleFSR2(
    const unsigned char* input, int inW, int inH,
    unsigned char* output, int outW, int outH,
    float sharpness,
    bool useRcas,
    bool rcasDenoise,
    float lfga,
    bool useTepd,
    float depth,
    bool useJitter,
    int numFrames,
    Fsr2JitterMode jitterMode,
    const std::set<int>& enabledPasses)
{
    // ── 1. Convert input to float ──
    size_t inPixels = (size_t)inW * inH;
    size_t outPixels = (size_t)outW * outH;

    std::vector<float> inputF(inPixels * 4);
    uint8ToFloat(input, inputF.data(), (int)inPixels);

    // ── 2. Compute render resolution ──
    // FSR2 upscales from render res to display res.
    // Render res = input res (the low-res image we want to upscale).
    // Display res = output res.
    int renderW = inW, renderH = inH;
    int displayW = outW, displayH = outH;

    // ── 3. Compute jitter sequence ──
    int phaseCount = fsr2GetJitterPhaseCount(renderW, displayW);
    if (!useJitter) phaseCount = 1; // Override: single phase = no jitter variation

    // If numFrames > phaseCount, we cycle the sequence
    int actualFrames = numFrames;
    if (actualFrames < 1) actualFrames = 1;

    std::cout << "  [FSR2] Render: " << renderW << "x" << renderH
              << " -> Display: " << displayW << "x" << displayH << std::endl;
    std::cout << "  [FSR2] Jitter: " << (useJitter ? "ON" : "OFF")
              << ", Frames: " << actualFrames
              << ", JitterPhaseCount: " << phaseCount << std::endl;
    std::cout << "  [FSR2] Enabled passes: ";
    for (int p : enabledPasses) std::cout << p << " ";
    std::cout << std::endl;

    // ── 4. Build synthetic flat depth buffer ──
    std::vector<float> depthBuf(inPixels, depth);

    // ── 5. Build synthetic zero motion vector buffer ──
    std::vector<float> mvBuf(inPixels * 2, 0.0f);

    // ── 6. Create FSR2 context ──
    Fsr2Context ctx;
    ctx.init(renderW, renderH, displayW, displayH);

    // ── 7. Prepare output buffer (float) ──
    std::vector<float> displayF(outPixels * 4, 0.0f);

    // ── 8. Jitter accumulation loop ──
    std::vector<float> jitteredFrame(inPixels * 4);
    std::vector<float> rcasOut(outPixels * 4);

    for (int frameIdx = 0; frameIdx < actualFrames; frameIdx++) {
        // Compute jitter offset for this frame
        float jX = 0.0f, jY = 0.0f;
        if (useJitter && phaseCount > 1) {
            fsr2GetJitterOffset(&jX, &jY, frameIdx, phaseCount);
        }

        // Generate the jittered low-res frame
        // The jitter shifts the source image by (jX, jY) pixels
        if (useJitter && (std::abs(jX) > 1e-6f || std::abs(jY) > 1e-6f)) {
            fsr2ResampleWithShift(inputF.data(), renderW, renderH,
                                   jX, jY,
                                   jitteredFrame.data(),
                                   jitterMode);
        } else {
            std::copy(inputF.begin(), inputF.end(), jitteredFrame.begin());
        }

        // Fill dispatch params
        Fsr2DispatchParams p;
        p.color          = jitteredFrame.data();
        p.depth          = depthBuf.data();
        p.motionVectors  = mvBuf.data();
        p.exposure       = nullptr;
        p.reactive       = nullptr;
        p.transparencyAndComposition = nullptr;
        p.renderW        = renderW;
        p.renderH        = renderH;
        p.displayW       = displayW;
        p.displayH       = displayH;
        p.jitterOffsetX  = jX;
        p.jitterOffsetY  = jY;
        p.motionVectorScaleX = 1.0f;
        p.motionVectorScaleY = 1.0f;
        p.frameTimeDelta = 16.667f;
        p.sharpness      = (useRcas && enabledPasses.count(FSR2_PASS_RCAS)) ? sharpness : 0.0f;
        p.enableSharpening = (useRcas && enabledPasses.count(FSR2_PASS_ACCUMULATE_SHARPEN));
        p.flags          = FSR2_ENABLE_AUTO_EXPOSURE;
        p.reset          = (frameIdx == 0); // Reset history on first frame

        // Run FSR2
        fsr2Dispatch(ctx, p, enabledPasses, displayF.data(), rcasDenoise);

        // Progress reporting for long accumulations
        if (actualFrames > 8 && (frameIdx + 1) % 8 == 0) {
            std::cout << "  [FSR2] Accumulated " << (frameIdx + 1) << "/" << actualFrames << " frames..." << std::endl;
        }
    }

    // ── 9. Apply post-processing (LFGA, TEPD) to final output ──
    for (int y = 0; y < displayH; y++) {
        for (int x = 0; x < displayW; x++) {
            size_t idx = ((size_t)y * displayW + x) * 4;
            float3 color(displayF[idx], displayF[idx+1], displayF[idx+2]);
            color = applyPostProcess(color, x, y, lfga, useTepd);
            displayF[idx+0] = color.x;
            displayF[idx+1] = color.y;
            displayF[idx+2] = color.z;
        }
    }

    // ── 10. Convert float output to uint8 ──
    floatToUint8(displayF.data(), output, (int)outPixels);
}
