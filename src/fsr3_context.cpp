// =============================================================================
// fsr3_context.cpp
// FSR 3.1.5 CPU Port — Main dispatch orchestration
// =============================================================================
#include "fsr3_context.h"
#include "fsr_math.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>

void fsr3Dispatch(
    Fsr3Context&            ctx,
    const Fsr3DispatchParams& params,
    const std::set<int>&    enabledPasses,
    float*                  outDisplayColor,
    bool                    useRcasDenoise)
{
    auto& buf = ctx.buf;
    int rW = params.renderW,  rH = params.renderH;
    int dW = params.displayW, dH = params.displayH;

    if (!ctx.initialized
        || buf.renderW  != rW || buf.renderH  != rH
        || buf.displayW != dW || buf.displayH != dH) {
        ctx.init(rW, rH, dW, dH);
    }
    if (params.reset) ctx.reset();

    buf.prevJitterX = buf.jitterX;
    buf.prevJitterY = buf.jitterY;
    buf.jitterX     = params.jitterOffsetX;
    buf.jitterY     = params.jitterOffsetY;

    bool autoExp = (params.flags & FSR3_ENABLE_AUTO_EXPOSURE) != 0;
    size_t rSize = (size_t)rW * rH;
    size_t dSize = (size_t)dW * dH;

    // ── Synthetic inputs for static image mode ────────────────────────────
    std::vector<float> synDepth, synMV, synReactive, synTc;
    const float* depthBuf    = params.depth;
    const float* mvBuf       = params.motionVectors;
    const float* reactiveBuf = params.reactive;
    const float* tcBuf       = params.transparencyAndComposition;

    if (!depthBuf)    { synDepth.assign(rSize, 0.5f);        depthBuf    = synDepth.data(); }
    if (!mvBuf)       { synMV.assign(rSize * 2, 0.0f);       mvBuf       = synMV.data(); }
    if (!reactiveBuf) { fsr3GenerateZeroReactiveMask(rW, rH, synReactive); reactiveBuf = synReactive.data(); }
    if (!tcBuf)       { synTc.assign(rSize, 0.0f);           tcBuf       = synTc.data(); }

    // ── Pass 10: Luminance Pyramid ────────────────────────────────────────
    if (enabledPasses.count(FSR3_PASS_COMPUTE_LUMINANCE_PYRAMID)) {
        fsr3PassComputeLuminancePyramid(params.color, buf, autoExp);
    } else {
        buf.autoExposure = 1.0f;
        std::copy(buf.luminanceCurrent.begin(), buf.luminanceCurrent.end(),
                  buf.luminancePrevious.begin());
        for (size_t i = 0; i < rSize; i++) {
            float r = params.color[i*4], g = params.color[i*4+1], b = params.color[i*4+2];
            buf.luminanceCurrent[i] = 0.2126f*r + 0.7152f*g + 0.0722f*b;
        }
        // Populate mip1 even if pass is disabled
        for (int my = 0; my < buf.mip1H; my++) {
            for (int mx = 0; mx < buf.mip1W; mx++) {
                int sx = std::min(mx*2, rW-1), sy = std::min(my*2, rH-1);
                buf.luminanceMip1[(size_t)my * buf.mip1W + mx] =
                    buf.luminanceCurrent[(size_t)sy * rW + sx];
            }
        }
    }

    // ── Pass 1: Reconstruct Previous Depth & Dilate MVs ──────────────────
    if (enabledPasses.count(FSR3_PASS_RECONSTRUCT_PREVIOUS_DEPTH)) {
        fsr3PassReconstructPreviousDepth(depthBuf, mvBuf, buf);
    } else {
        for (size_t i = 0; i < rSize; i++) {
            buf.dilatedDepth[i]          = depthBuf[i];
            buf.dilatedMotionVectorsX[i] = mvBuf[i*2];
            buf.dilatedMotionVectorsY[i] = mvBuf[i*2+1];
        }
    }

    // ── Pass 0: Depth Clip ────────────────────────────────────────────────
    if (enabledPasses.count(FSR3_PASS_DEPTH_CLIP)) {
        fsr3PassDepthClip(depthBuf, buf);
    } else {
        std::fill(buf.disocclusionMask.begin(), buf.disocclusionMask.end(), 1.0f);
    }

    // ── Pass 3: Prepare Reactivity ────────────────────────────────────────
    if (enabledPasses.count(FSR3_PASS_PREPARE_REACTIVITY)) {
        fsr3PassPrepareReactivity(reactiveBuf, tcBuf, buf);
    } else {
        std::fill(buf.preparedReactivity.begin(), buf.preparedReactivity.end(), 0.0f);
    }

    // ── Pass 2: Lock (at display res) ─────────────────────────────────────
    if (enabledPasses.count(FSR3_PASS_LOCK)) {
        fsr3PassLock(params.color, buf);
    } else {
        std::fill(buf.lockMask.begin(), buf.lockMask.end(), 0.0f);
    }

    // ── Pass 9: New Locks ─────────────────────────────────────────────────
    if (enabledPasses.count(FSR3_PASS_NEW_LOCKS)) {
        fsr3PassNewLocks(params.color, buf);
    } else {
        std::fill(buf.newLockMask.begin(), buf.newLockMask.end(), 0.0f);
    }

    // ── Passes 4/5: Accumulate ────────────────────────────────────────────
    bool hasAccum   = enabledPasses.count(FSR3_PASS_ACCUMULATE)         > 0;
    bool hasSharpen = enabledPasses.count(FSR3_PASS_ACCUMULATE_SHARPEN) > 0;
    bool useAccum   = hasAccum || hasSharpen;

    if (useAccum) {
        bool builtinSharpen = hasSharpen && !hasAccum && params.enableSharpening;
        fsr3PassAccumulate(params.color, buf, params.sharpness, builtinSharpen);
    }

    // ── Pass 6: RCAS ──────────────────────────────────────────────────────
    if (enabledPasses.count(FSR3_PASS_RCAS) && useAccum) {
        std::vector<float> rcasIn(buf.internalColorHistory);
        fsr3PassRCAS(rcasIn.data(), dW, dH,
                     buf.internalColorHistory.data(),
                     params.sharpness, useRcasDenoise);
    }

    // ── Write final output ────────────────────────────────────────────────
    if (useAccum) {
        std::memcpy(outDisplayColor, buf.internalColorHistory.data(),
                    dSize * 4 * sizeof(float));
    } else {
        // No accumulate ran — bilinear passthrough
        float sx = (float)rW / dW, sy = (float)rH / dH;
        for (int oy = 0; oy < dH; oy++) {
            for (int ox = 0; ox < dW; ox++) {
                int x0    = std::max(0, std::min(rW-1, (int)(((float)ox+0.5f)*sx - 0.5f)));
                int y0    = std::max(0, std::min(rH-1, (int)(((float)oy+0.5f)*sy - 0.5f)));
                size_t si = ((size_t)y0 * rW + x0) * 4;
                size_t di = ((size_t)oy * dW + ox) * 4;
                outDisplayColor[di+0] = params.color[si+0];
                outDisplayColor[di+1] = params.color[si+1];
                outDisplayColor[di+2] = params.color[si+2];
                outDisplayColor[di+3] = params.color[si+3];
            }
        }
    }

    buf.frameIndex++;
    buf.firstFrame = false;
}
