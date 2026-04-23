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

    if (!ctx.initialized || buf.renderW != rW || buf.renderH != rH ||
        buf.displayW != dW || buf.displayH != dH) {
        ctx.init(rW, rH, dW, dH);
    }

    if (params.reset) {
        ctx.reset();
    }

    // Advance jitter tracking: save previous, store current
    buf.prevJitterX = buf.jitterX;
    buf.prevJitterY = buf.jitterY;
    buf.jitterX = params.jitterOffsetX;
    buf.jitterY = params.jitterOffsetY;

    bool autoExp = (params.flags & FSR2_ENABLE_AUTO_EXPOSURE) != 0;

    size_t rSize = (size_t)rW * rH;
    size_t dSize = (size_t)dW * dH;

    // ── Synthetic inputs for static image mode ────────────────────────────
    // Depth: uniform flat plane (user-specified, default 0.5)
    std::vector<float> syntheticDepth;
    const float* depthBuf = params.depth;
    if (!depthBuf) {
        syntheticDepth.assign(rSize, 0.5f);
        depthBuf = syntheticDepth.data();
    }

    // Motion vectors: zero (static scene).
    // The jitter-delta compensation is handled INSIDE fsr2PassAccumulate
    // by reading buf.jitterX/prevJitterX, so we do not add it here.
    // If a 3D app supplies real motion vectors, those are used as-is.
    std::vector<float> syntheticMV;
    const float* mvBuf = params.motionVectors;
    if (!mvBuf) {
        syntheticMV.assign(rSize * 2, 0.0f);
        mvBuf = syntheticMV.data();
    }

    // Reactive: zero mask (no transparent/reactive objects in static images)
    std::vector<float> syntheticReactive;
    const float* reactiveBuf = params.reactive;
    if (!reactiveBuf) {
        fsr2GenerateZeroReactiveMask(rW, rH, syntheticReactive);
        reactiveBuf = syntheticReactive.data();
    }

    // ── Pass 6: Compute Luminance Pyramid ─────────────────────────────────
    if (enabledPasses.count(FSR2_PASS_COMPUTE_LUMINANCE_PYRAMID)) {
        fsr2PassComputeLuminancePyramid(params.color, buf, autoExp);
    } else {
        buf.autoExposure = 1.0f;
        // Still need to populate luminanceCurrent for the lock pass
        // (it uses per-pixel luma, not the auto-exposure scalar)
        int count = rW * rH;
        std::copy(buf.luminanceCurrent.begin(), buf.luminanceCurrent.end(),
                  buf.luminancePrevious.begin());
        for (int i = 0; i < count; i++) {
            float r = params.color[i * 4 + 0];
            float g = params.color[i * 4 + 1];
            float b = params.color[i * 4 + 2];
            buf.luminanceCurrent[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }
    }

    // ── Pass 1: Reconstruct Previous Depth & Dilate Motion Vectors ────────
    if (enabledPasses.count(FSR2_PASS_RECONSTRUCT_PREVIOUS_DEPTH)) {
        fsr2PassReconstructPreviousDepth(depthBuf, mvBuf, buf);
    } else {
        // Neutral: copy depth as-is, use provided motion vectors unchanged
        for (size_t i = 0; i < rSize; i++) {
            buf.dilatedDepth[i] = depthBuf[i];
            buf.dilatedMotionVectorsX[i] = mvBuf[i * 2 + 0];
            buf.dilatedMotionVectorsY[i] = mvBuf[i * 2 + 1];
        }
    }

    // ── Pass 0: Depth Clip (Disocclusion Detection) ────────────────────────
    if (enabledPasses.count(FSR2_PASS_DEPTH_CLIP)) {
        fsr2PassDepthClip(depthBuf, buf);
    } else {
        // Neutral: all pixels have valid history (no disocclusion)
        std::fill(buf.disocclusionMask.begin(), buf.disocclusionMask.end(), 1.0f);
    }

    // ── Pass 2: Lock ───────────────────────────────────────────────────────
    if (enabledPasses.count(FSR2_PASS_LOCK)) {
        fsr2PassLock(params.color, buf);
    } else {
        // Neutral: no locks (maximum temporal freedom for all pixels)
        std::fill(buf.lockMask.begin(), buf.lockMask.end(), 0.0f);
    }

    // ── Pass 3 or 4: Accumulate ────────────────────────────────────────────
    bool hasAccumPass   = enabledPasses.count(FSR2_PASS_ACCUMULATE) > 0;
    bool hasSharpenPass = enabledPasses.count(FSR2_PASS_ACCUMULATE_SHARPEN) > 0;
    bool useAccum = hasAccumPass || hasSharpenPass;

    if (useAccum) {
        // Pass 3 = accumulate only. Pass 4 = accumulate with built-in sharpen.
        // If only pass 4 is listed (not pass 3), use the sharpen variant.
        bool builtinSharpen = hasSharpenPass && !hasAccumPass && params.enableSharpening;
        fsr2PassAccumulate(params.color, reactiveBuf, buf, params.sharpness, builtinSharpen);
    }

    // ── Pass 5: RCAS ───────────────────────────────────────────────────────
    bool runRCAS = enabledPasses.count(FSR2_PASS_RCAS) > 0 && params.sharpness > 0.0f;
    if (runRCAS) {
        // RCAS reads from internalColorHistory and writes back to it
        std::vector<float> rcasInput(buf.internalColorHistory);
        fsr2PassRCAS(rcasInput.data(), dW, dH,
                     buf.internalColorHistory.data(),
                     params.sharpness, useRcasDenoise);
    }

    // ── Write final output ─────────────────────────────────────────────────
    if (useAccum) {
        std::memcpy(outDisplayColor, buf.internalColorHistory.data(),
                    dSize * 4 * sizeof(float));
    } else {
        // No accumulate ran — bilinear upscale as passthrough
        float sx = (float)rW / (float)dW;
        float sy = (float)rH / (float)dH;
        for (int oy = 0; oy < dH; oy++) {
            for (int ox = 0; ox < dW; ox++) {
                float fx = ((float)ox + 0.5f) * sx - 0.5f;
                float fy = ((float)oy + 0.5f) * sy - 0.5f;
                int x0 = std::max(0, std::min(rW - 1, (int)fx));
                int y0 = std::max(0, std::min(rH - 1, (int)fy));
                size_t sIdx = ((size_t)y0 * rW + x0) * 4;
                size_t dIdx = ((size_t)oy * dW + ox) * 4;
                outDisplayColor[dIdx+0] = params.color[sIdx+0];
                outDisplayColor[dIdx+1] = params.color[sIdx+1];
                outDisplayColor[dIdx+2] = params.color[sIdx+2];
                outDisplayColor[dIdx+3] = params.color[sIdx+3];
            }
        }
    }

    // ── Advance frame counter ──────────────────────────────────────────────
    buf.frameIndex++;
    buf.firstFrame = false;
}
