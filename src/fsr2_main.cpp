// =============================================================================
// fsr2_main.cpp
// FSR 2.3.4 CPU Port — Static image upscaling entry point
//
// DATA FLOW:
//   The input PNG is the "render resolution" image. FSR2 treats it as if
//   a game engine rendered at this resolution. We do NOT downscale it.
//   FSR2 upscales from (inW x inH) to (outW x outH) using temporal
//   accumulation of synthetic sub-pixel jitter frames.
//
//   Frame generation:
//     1. Start with input PNG at render resolution (inW x inH).
//     2. For each jitter frame, shift by a sub-pixel Halton offset.
//        When jitter is OFF, all frames receive the unshifted image.
//     3. Run all enabled FSR2 passes on each frame.
//     4. After N frames, the accumulated result is the upscaled output.
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

static void uint8ToFloat(const unsigned char* src, float* dst, int pixelCount) {
    for (int i = 0; i < pixelCount * 4; i++)
        dst[i] = src[i] / 255.0f;
}
static void floatToUint8(const float* src, unsigned char* dst, int pixelCount) {
    for (int i = 0; i < pixelCount * 4; i++)
        dst[i] = (unsigned char)(clamp(src[i], 0.0f, 1.0f) * 255.0f + 0.5f);
}

void scaleFSR2(
    const unsigned char* input, int inW, int inH,
    unsigned char* output, int outW, int outH,
    float sharpness,        // RCAS sharpness in stops: 0.0=max, 0.2=default, 2.0=min
    bool useRcas,           // whether RCAS (pass 5) runs at all
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

    std::vector<float> inputF(renderPixels * 4);
    uint8ToFloat(input, inputF.data(), (int)renderPixels);

    int phaseCount   = useJitter ? fsr2GetJitterPhaseCount(renderW, displayW) : 1;
    int actualFrames = std::max(1, numFrames);

    std::cout << "  [FSR2] Render: " << renderW << "x" << renderH
              << " -> Display: " << displayW << "x" << displayH << std::endl;
    std::cout << "  [FSR2] Jitter: " << (useJitter ? "ON" : "OFF")
              << "  Frames: " << actualFrames
              << "  PhaseCount: " << phaseCount << std::endl;
    std::cout << "  [FSR2] RCAS: " << (useRcas ? "ON" : "OFF");
    if (useRcas)
        std::cout << "  Sharpness: " << sharpness
                  << " (exp2(-" << sharpness << ")=" << std::exp2(-sharpness) << " multiplier)";
    std::cout << std::endl;
    std::cout << "  [FSR2] Enabled passes: ";
    for (int p : enabledPasses) std::cout << p << " ";
    std::cout << std::endl;

    std::vector<float> depthBuf(renderPixels, depth);
    std::vector<float> mvBuf(renderPixels * 2, 0.0f);

    Fsr2Context ctx;
    ctx.init(renderW, renderH, displayW, displayH);

    std::vector<float> displayF(displayPixels * 4, 0.0f);
    std::vector<float> jitteredFrame(renderPixels * 4);

    for (int frameIdx = 0; frameIdx < actualFrames; frameIdx++) {
        float jX = 0.0f, jY = 0.0f;
        if (useJitter) {
            fsr2GetJitterOffset(&jX, &jY, frameIdx, phaseCount);
        }

        if (useJitter && (std::abs(jX) > 1e-7f || std::abs(jY) > 1e-7f)) {
            fsr2ResampleWithShift(inputF.data(), renderW, renderH,
                                   jX, jY,
                                   jitteredFrame.data(),
                                   jitterMode);
        } else {
            std::copy(inputF.begin(), inputF.end(), jitteredFrame.begin());
        }

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

        // Always pass sharpness as-is. sharpness=0.0 means maximum sharpening.
        // Whether RCAS runs at all is determined by whether pass 5 is in
        // enabledPasses (controlled by --fsr2-rcas on/off in main.cpp).
        p.sharpness        = sharpness;
        p.enableSharpening = enabledPasses.count(FSR2_PASS_ACCUMULATE_SHARPEN) > 0;
        p.flags            = 0; // no FSR2_ENABLE_AUTO_EXPOSURE for SDR
        p.reset            = (frameIdx == 0);

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

    floatToUint8(displayF.data(), output, (int)displayPixels);
}
