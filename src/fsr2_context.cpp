// =============================================================================
// fsr2_context.cpp
// FSR 2.3.4 CPU Port — Main dispatch orchestration
// =============================================================================
#include "fsr2_context.h"
#include "fsr_math.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>

void fsr2Dispatch(
    Fsr2Context& ctx,
    const Fsr2DispatchParams& params,
    const std::set<int>& enabledPasses,
    float* outDisplayColor,
    bool useRcasDenoise)
{
    auto& buf = ctx.buf;
    int rW = params.renderW, rH = params.renderH;
    int dW = params.displayW, dH = params.displayH;

    // Validate/reinit if dimensions changed
    if (!ctx.initialized || buf.renderW != rW || buf.renderH != rH ||
        buf.displayW != dW || buf.displayH != dH) {
        ctx.init(rW, rH, dW, dH);
    }

    // Handle reset (camera cut)
    if (params.reset) {
        ctx.reset();
    }

    // Store jitter for this frame
    buf.jitterX = params.jitterOffsetX;
    buf.jitterY = params.jitterOffsetY;

    bool autoExp = (params.flags & FSR2_ENABLE_AUTO_EXPOSURE) != 0;

    // ── Build synthetic inputs for static image adaptation ──
    size_t rSize = (size_t)rW * rH;
    size_t dSize = (size_t)dW * dH;

    // Depth buffer: uniform flat plane (0.5) if not provided
    std::vector<float> syntheticDepth;
    const float* depthBuf = params.depth;
    if (!depthBuf) {
        syntheticDepth.assign(rSize, 0.5f);
        depthBuf = syntheticDepth.data();
    }

    // Motion vectors: zero (static scene) if not provided
    std::vector<float> syntheticMV;
    const float* mvBuf = params.motionVectors;
    if (!mvBuf) {
        syntheticMV.assign(rSize * 2, 0.0f);
        mvBuf = syntheticMV.data();
    }

    // Reactive mask: zero if not provided
    std::vector<float> syntheticReactive;
    const float* reactiveBuf = params.reactive;
    if (!reactiveBuf) {
        fsr2GenerateZeroReactiveMask(rW, rH, syntheticReactive);
        reactiveBuf = syntheticReactive.data();
    }

    // ── Pass 6: Compute Luminance Pyramid ──
    if (enabledPasses.count(FSR2_PASS_COMPUTE_LUMINANCE_PYRAMID)) {
        fsr2PassComputeLuminancePyramid(params.color, buf, autoExp);
    } else {
        buf.autoExposure = 1.0f;
    }

    // ── Pass 1: Reconstruct Previous Depth & Dilate Motion Vectors ──
    if (enabledPasses.count(FSR2_PASS_RECONSTRUCT_PREVIOUS_DEPTH)) {
        fsr2PassReconstructPreviousDepth(depthBuf, mvBuf, buf);
    } else {
        // Neutral state: no dilation, uniform depth, zero MVs
        std::fill(buf.dilatedDepth.begin(), buf.dilatedDepth.end(), 0.5f);
        std::fill(buf.dilatedMotionVectorsX.begin(), buf.dilatedMotionVectorsX.end(), 0.0f);
        std::fill(buf.dilatedMotionVectorsY.begin(), buf.dilatedMotionVectorsY.end(), 0.0f);
    }

    // ── Pass 0: Depth Clip (Disocclusion Detection) ──
    if (enabledPasses.count(FSR2_PASS_DEPTH_CLIP)) {
        fsr2PassDepthClip(depthBuf, buf);
    } else {
        // Neutral: no disocclusion (all pixels have valid history)
        std::fill(buf.disocclusionMask.begin(), buf.disocclusionMask.end(), 1.0f);
    }

    // ── Pass 2: Lock ──
    if (enabledPasses.count(FSR2_PASS_LOCK)) {
        fsr2PassLock(params.color, buf);
    } else {
        // Neutral: no locks (maximum temporal freedom)
        std::fill(buf.lockMask.begin(), buf.lockMask.end(), 0.0f);
    }

    // ── Pass 7/8: Reactive/TCR Mask Generation (optional) ──
    // These run before accumulate as they feed into it.
    // For static images, we use the pre-computed zero mask.
    // (Users can supply a pre-computed reactive mask via params.reactive)

    // ── Pass 3 or 4: Accumulate ──
    bool useSharpenPass = enabledPasses.count(FSR2_PASS_ACCUMULATE_SHARPEN) &&
                         !enabledPasses.count(FSR2_PASS_ACCUMULATE) &&
                         params.enableSharpening;
    bool useAccumPass   = enabledPasses.count(FSR2_PASS_ACCUMULATE) ||
                         enabledPasses.count(FSR2_PASS_ACCUMULATE_SHARPEN);

    if (useAccumPass) {
        bool builtinSharpen = enabledPasses.count(FSR2_PASS_ACCUMULATE_SHARPEN) &&
                              params.enableSharpening;
        fsr2PassAccumulate(params.color, reactiveBuf, buf, params.sharpness, builtinSharpen);
    }

    // ── Pass 5: RCAS ──
    bool runRCAS = enabledPasses.count(FSR2_PASS_RCAS) && params.sharpness > 0.0f;

    if (runRCAS) {
        // RCAS operates on the output of the accumulate pass
        std::vector<float> rcasInput(buf.internalColorHistory);
        fsr2PassRCAS(rcasInput.data(), dW, dH,
                     buf.internalColorHistory.data(),
                     params.sharpness, useRcasDenoise);
    }

    // ── Write final output ──
    if (useAccumPass) {
        std::memcpy(outDisplayColor, buf.internalColorHistory.data(), dSize * 4 * sizeof(float));
    } else {
        // No accumulate ran — output the current color upscaled via simple bilinear
        float scaleX = (float)rW / (float)dW;
        float scaleY = (float)rH / (float)dH;
        for (int dy = 0; dy < dH; dy++) {
            for (int dx = 0; dx < dW; dx++) {
                float srcX = ((float)dx + 0.5f) * scaleX - 0.5f;
                float srcY = ((float)dy + 0.5f) * scaleY - 0.5f;
                int x0 = std::max(0, std::min(rW-1, (int)srcX));
                int y0 = std::max(0, std::min(rH-1, (int)srcY));
                size_t sIdx = ((size_t)y0 * rW + x0) * 4;
                size_t dIdx = ((size_t)dy * dW + dx) * 4;
                outDisplayColor[dIdx+0] = params.color[sIdx+0];
                outDisplayColor[dIdx+1] = params.color[sIdx+1];
                outDisplayColor[dIdx+2] = params.color[sIdx+2];
                outDisplayColor[dIdx+3] = params.color[sIdx+3];
            }
        }
    }

    // Advance frame counter
    buf.frameIndex++;
    buf.firstFrame = false;
}
