// =============================================================================
// fsr2_main.cpp
// FSR 2.3.4 CPU Port — Static image upscaling entry point
//
// DATA FLOW CLARIFICATION:
//   The input PNG is the "render resolution" image — FSR2 treats it as if
//   a game engine rendered at this resolution. We do NOT downscale it.
//   FSR2 upscales from (inW x inH) to (outW x outH) using temporal
//   accumulation of synthetic sub-pixel jitter frames.
//
//   Frame generation:
//     1. Start with the full input PNG at render resolution (inW x inH).
//     2. For each jitter frame, shift the image by a sub-pixel Halton offset.
//        (When jitter is OFF, all frames receive the unshifted image.)
//     3. Run all FSR2 passes on each shifted frame.
//     4. After N frames, the accumulated result is the upscaled output.
//
//   Why multiple frames?
//     Each jitter frame samples a slightly different sub-pixel position.
//     FSR2's accumulate pass combines them all, effectively reconstructing
//     more detail than any single frame contains. This is temporal super-sampling.
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

// ── uint8 ↔ float ────────────────────────────────────────────────────────────
static void uint8ToFloat(const unsigned char* src, float* dst, int pixelCount) {
    for (int i = 0; i < pixelCount * 4; i++)
        dst[i] = src[i] / 255.0f;
}
static void floatToUint8(const float* src, unsigned char* dst, int pixelCount) {
    for (int i = 0; i < pixelCount * 4; i++)
        dst[i] = (unsigned char)(clamp(src[i], 0.0f, 1.0f) * 255.0f + 0.5f);
}

// ── Bicubic downscale for float RGBA ─────────────────────────────────────────
// Used only when the user requests a render-res smaller than the input
// (not the normal case — normally renderW == inW).
static float cubicWF(float t) {
    t = std::abs(t);
    if (t < 1.0f) return  1.5f*t*t*t - 2.5f*t*t + 1.0f;
    if (t < 2.0f) return -0.5f*t*t*t + 2.5f*t*t - 4.0f*t + 2.0f;
    return 0.0f;
}

static void bicubicResampleF(const float* src, int srcW, int srcH,
                              float* dst, int dstW, int dstH)
{
    float scX = (float)srcW / (float)dstW;
    float scY = (float)srcH / (float)dstH;
    float fscX = std::min(1.0f, (float)dstW / (float)srcW);
    float fscY = std::min(1.0f, (float)dstH / (float)srcH);
    float radX = 2.0f / fscX, radY = 2.0f / fscY;

    for (int dy = 0; dy < dstH; dy++) {
        float sY = ((float)dy + 0.5f) / scY - 0.5f;
        int y0 = std::max(0, (int)std::floor(sY - radY + 1));
        int y1 = std::min(srcH - 1, (int)std::floor(sY + radY));
        for (int dx = 0; dx < dstW; dx++) {
            float sX = ((float)dx + 0.5f) / scX - 0.5f;
            int x0 = std::max(0, (int)std::floor(sX - radX + 1));
            int x1 = std::min(srcW - 1, (int)std::floor(sX + radX));
            double r=0,g=0,b=0,a=0,ws=0;
            for (int cy = y0; cy <= y1; cy++) {
                float wy = cubicWF((sY - cy) * fscY);
                for (int cx = x0; cx <= x1; cx++) {
                    float w = cubicWF((sX - cx) * fscX) * wy;
                    size_t i = ((size_t)cy * srcW + cx) * 4;
                    r += src[i+0]*w; g += src[i+1]*w;
                    b += src[i+2]*w; a += src[i+3]*w;
                    ws += w;
                }
            }
            if (ws < 1e-9) ws = 1.0;
            size_t di = ((size_t)dy * dstW + dx) * 4;
            dst[di+0] = clamp((float)(r/ws), 0.0f, 1.0f);
            dst[di+1] = clamp((float)(g/ws), 0.0f, 1.0f);
            dst[di+2] = clamp((float)(b/ws), 0.0f, 1.0f);
            dst[di+3] = clamp((float)(a/ws), 0.0f, 1.0f);
        }
    }
}

// =============================================================================
// scaleFSR2 — Main entry point
//
// input/output: uint8 RGBA
// inW,inH: source (render) resolution
// outW,outH: destination (display) resolution
// =============================================================================
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
    int renderW  = inW,  renderH  = inH;
    int displayW = outW, displayH = outH;
    size_t renderPixels  = (size_t)renderW  * renderH;
    size_t displayPixels = (size_t)displayW * displayH;

    // Convert input to float
    std::vector<float> inputF(renderPixels * 4);
    uint8ToFloat(input, inputF.data(), (int)renderPixels);

    // Jitter sequence setup
    int phaseCount = useJitter ? fsr2GetJitterPhaseCount(renderW, displayW) : 1;
    int actualFrames = std::max(1, numFrames);

    std::cout << "  [FSR2] Render: " << renderW << "x" << renderH
              << " -> Display: " << displayW << "x" << displayH << std::endl;
    std::cout << "  [FSR2] Jitter: " << (useJitter ? "ON" : "OFF")
              << "  Frames: " << actualFrames
              << "  PhaseCount: " << phaseCount << std::endl;
    std::cout << "  [FSR2] Enabled passes: ";
    for (int p : enabledPasses) std::cout << p << " ";
    std::cout << std::endl;

    // Flat depth buffer (uniform scene depth)
    std::vector<float> depthBuf(renderPixels, depth);

    // Zero motion vectors (static scene — jitter delta handled inside accumulate)
    std::vector<float> mvBuf(renderPixels * 2, 0.0f);

    // FSR2 context
    Fsr2Context ctx;
    ctx.init(renderW, renderH, displayW, displayH);

    // Output buffer (float)
    std::vector<float> displayF(displayPixels * 4, 0.0f);

    // Jittered frame buffer (render res, float)
    std::vector<float> jitteredFrame(renderPixels * 4);

    for (int frameIdx = 0; frameIdx < actualFrames; frameIdx++) {
        // Compute Halton jitter offset for this frame
        float jX = 0.0f, jY = 0.0f;
        if (useJitter) {
            fsr2GetJitterOffset(&jX, &jY, frameIdx, phaseCount);
        }
        // When jitter is OFF: jX=jY=0 every frame.
        // The jitter delta in fsr2_pass_accumulate will be (0-0)=0, which
        // means the reprojection is a straight look-up with zero offset —
        // correct for a static-scene, no-jitter accumulation.

        // Generate the jittered frame by shifting the source image
        if (useJitter && (std::abs(jX) > 1e-7f || std::abs(jY) > 1e-7f)) {
            fsr2ResampleWithShift(inputF.data(), renderW, renderH,
                                   jX, jY,
                                   jitteredFrame.data(),
                                   jitterMode);
        } else {
            std::copy(inputF.begin(), inputF.end(), jitteredFrame.begin());
        }

        // Build dispatch params
        Fsr2DispatchParams p;
        p.color          = jitteredFrame.data();
        p.depth          = depthBuf.data();
        p.motionVectors  = mvBuf.data();   // zero — jitter delta added inside accumulate
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
        p.enableSharpening = enabledPasses.count(FSR2_PASS_ACCUMULATE_SHARPEN) > 0;
        // Do not set FSR2_ENABLE_AUTO_EXPOSURE for SDR images.
        // autoExposure stays 1.0, preventing the darkening bug.
        p.flags          = 0;
        // Reset history only on the very first frame
        p.reset          = (frameIdx == 0);

        fsr2Dispatch(ctx, p, enabledPasses, displayF.data(), rcasDenoise);

        if (actualFrames > 8 && (frameIdx + 1) % 8 == 0) {
            std::cout << "  [FSR2] Accumulated " << (frameIdx + 1)
                      << "/" << actualFrames << " frames..." << std::endl;
        }
    }

    // Apply post-processing (LFGA, TEPD) to final output
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

    // Convert float to uint8
    floatToUint8(displayF.data(), output, (int)displayPixels);
}
