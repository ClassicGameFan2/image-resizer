// =============================================================================
// fsr3_types.h
// FSR 3.1.5 CPU Port — Shared types, constants, and internal resource buffers
// Based on: ffx_fsr3upscaler.h, ffx_fsr3upscaler_common.h (AMD MIT)
//
// KEY FSR 3.1 UPSCALER DIFFERENCES FROM FSR 2.3.4:
//   1. Lock pass now runs at DISPLAY resolution (not render resolution).
//      This saves ALU in the accumulation pass and avoids precision issues.
//   2. New "prepare reactivity" pass (pass 3) combines dilated T&C mask
//      and motion divergence into a single reactivity buffer.
//   3. Shading change detection is improved via luma instability tracking.
//   4. Accumulation factor (convergence) is tracked per-pixel in a separate
//      buffer rather than being derived purely from the frame index.
//   5. disocclusionFactor uses a 2x2 depth comparison (not single pixel).
//   6. fMinDisocclusionAccumulation = -0.333 (new default, reduces ghosting).
//   7. RCAS 3.1.5: lowerLimiterMultiplier re-introduced in RCAS
//      (fix for possible negative RCAS output — same fix as FSR 1.2.2).
//   8. New locks computed at display res by a dedicated pass (pass 7).
// =============================================================================
#pragma once
#include "fsr_math.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

// ----------------------------------------------------------------
// FSR3 Upscaler Pass IDs (mirrors FfxFsr3UpscalerPass)
// ----------------------------------------------------------------
#define FSR3_PASS_DEPTH_CLIP                         0
#define FSR3_PASS_RECONSTRUCT_PREVIOUS_DEPTH         1
#define FSR3_PASS_LOCK                               2
#define FSR3_PASS_PREPARE_REACTIVITY                 3
#define FSR3_PASS_ACCUMULATE                         4
#define FSR3_PASS_ACCUMULATE_SHARPEN                 5
#define FSR3_PASS_RCAS                               6
#define FSR3_PASS_GENERATE_REACTIVE                  7
#define FSR3_PASS_TCR_AUTOGENERATE                   8
#define FSR3_PASS_NEW_LOCKS                          9
#define FSR3_PASS_COMPUTE_LUMINANCE_PYRAMID          10
#define FSR3_PASS_COUNT                              11

// ----------------------------------------------------------------
// FSR3 Context Flags
// ----------------------------------------------------------------
#define FSR3_ENABLE_DEPTH_INVERTED                   (1 << 0)
#define FSR3_ENABLE_DEPTH_INFINITE                   (1 << 1)
#define FSR3_ENABLE_AUTO_EXPOSURE                    (1 << 2)
#define FSR3_ENABLE_HIGH_DYNAMIC_RANGE               (1 << 3)
#define FSR3_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION (1 << 4)
#define FSR3_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS  (1 << 5)

// ----------------------------------------------------------------
// Internal resource buffers.
// All color stored as float linear [0..1] internally.
// ----------------------------------------------------------------
struct Fsr3InternalBuffers {
    int renderW  = 0, renderH  = 0;
    int displayW = 0, displayH = 0;

    // Pass 10: Luminance pyramid (render res, 1 channel)
    std::vector<float> luminanceCurrent;     // renderW * renderH
    std::vector<float> luminancePrevious;    // renderW * renderH

    // Downsampled luminance mip (1/2 render res, 1 channel)
    // Used by accumulate pass for shading change detection.
    std::vector<float> luminanceMip1;        // (renderW/2) * (renderH/2)
    int                mip1W = 0, mip1H = 0;

    float autoExposure     = 1.0f;
    float prevAutoExposure = 1.0f;

    // Pass 1: Dilated motion vectors (render res, float XY)
    std::vector<float> dilatedMotionVectorsX; // renderW * renderH
    std::vector<float> dilatedMotionVectorsY; // renderW * renderH

    // Pass 1: Dilated depth (render res, float)
    std::vector<float> dilatedDepth;          // renderW * renderH

    // Pass 0: Disocclusion mask (render res, float)
    // 1.0 = visible/occluded (history valid), 0.0 = disoccluded
    // FSR3.1: Computed from 2x2 depth comparison (vs single pixel in FSR2).
    std::vector<float> disocclusionMask;      // renderW * renderH

    // Pass 3: Prepare reactivity — combined dilated reactive (render res)
    // max(dilatedTcMask, motionDivergence)
    std::vector<float> preparedReactivity;    // renderW * renderH

    // Pass 2: Lock mask at DISPLAY resolution (FSR3.1 change from FSR2!)
    // In FSR2 lock was at render res. FSR3.1 moves it to display res.
    std::vector<float> lockMask;              // displayW * displayH

    // Pass 9: New locks at display resolution
    // New locks from current frame's neighborhood stability analysis.
    std::vector<float> newLockMask;           // displayW * displayH

    // Lock luminance (render res, for shading change detection)
    std::vector<float> lockLuminance;         // renderW * renderH

    // Luma instability (display res, float)
    // Tracks how much the upscaled luma changes frame-to-frame.
    std::vector<float> lumaInstability;       // displayW * displayH

    // Accumulation factor (display res, float [0..1])
    // Tracks per-pixel convergence. Replaces the pure frame-index
    // exponential used in FSR2. Allows faster convergence after
    // disocclusion by resetting only affected pixels.
    std::vector<float> accumulationFactor;    // displayW * displayH
    std::vector<float> prevAccumulationFactor;// displayW * displayH

    // Pass 4: Accumulated color at display res (float4 RGBA LINEAR)
    // History stored in LINEAR space (same as FSR2 CPU port design).
    std::vector<float> accumulatedColor;      // displayW * displayH * 4
    std::vector<float> prevAccumulatedColor;  // displayW * displayH * 4

    // Final output ready for RCAS or direct output
    std::vector<float> internalColorHistory;  // displayW * displayH * 4

    // Frame state
    int   frameIndex  = 0;
    bool  firstFrame  = true;
    float jitterX     = 0.0f, jitterY     = 0.0f;
    float prevJitterX = 0.0f, prevJitterY = 0.0f;

    void init(int rW, int rH, int dW, int dH) {
        renderW  = rW;  renderH  = rH;
        displayW = dW;  displayH = dH;
        mip1W    = std::max(1, rW / 2);
        mip1H    = std::max(1, rH / 2);

        size_t rSize = (size_t)rW * rH;
        size_t dSize = (size_t)dW * dH;
        size_t m1Size = (size_t)mip1W * mip1H;

        luminanceCurrent.assign(rSize, 0.0f);
        luminancePrevious.assign(rSize, 0.0f);
        luminanceMip1.assign(m1Size, 0.0f);

        dilatedMotionVectorsX.assign(rSize, 0.0f);
        dilatedMotionVectorsY.assign(rSize, 0.0f);
        dilatedDepth.assign(rSize, 0.5f);
        disocclusionMask.assign(rSize, 1.0f);
        preparedReactivity.assign(rSize, 0.0f);

        lockMask.assign(dSize, 0.0f);
        newLockMask.assign(dSize, 0.0f);
        lockLuminance.assign(rSize, 0.0f);

        lumaInstability.assign(dSize, 0.0f);
        accumulationFactor.assign(dSize, 0.0f);
        prevAccumulationFactor.assign(dSize, 0.0f);

        accumulatedColor.assign(dSize * 4, 0.0f);
        prevAccumulatedColor.assign(dSize * 4, 0.0f);
        internalColorHistory.assign(dSize * 4, 0.0f);

        autoExposure = prevAutoExposure = 1.0f;
        frameIndex = 0; firstFrame = true;
        jitterX = jitterY = prevJitterX = prevJitterY = 0.0f;
    }

    void reset() {
        std::fill(accumulatedColor.begin(),       accumulatedColor.end(),       0.0f);
        std::fill(prevAccumulatedColor.begin(),   prevAccumulatedColor.end(),   0.0f);
        std::fill(internalColorHistory.begin(),   internalColorHistory.end(),   0.0f);
        std::fill(lockMask.begin(),               lockMask.end(),               0.0f);
        std::fill(newLockMask.begin(),            newLockMask.end(),            0.0f);
        std::fill(lumaInstability.begin(),        lumaInstability.end(),        0.0f);
        std::fill(accumulationFactor.begin(),     accumulationFactor.end(),     0.0f);
        std::fill(prevAccumulationFactor.begin(), prevAccumulationFactor.end(), 0.0f);
        frameIndex = 0; firstFrame = true;
        prevJitterX = prevJitterY = 0.0f;
    }
};

// ----------------------------------------------------------------
// FSR3 Dispatch Parameters
// ----------------------------------------------------------------
struct Fsr3DispatchParams {
    const float* color         = nullptr; // render res, float RGBA linear
    const float* depth         = nullptr; // render res, float [0..1]
    const float* motionVectors = nullptr; // render res, float XY pairs
    const float* exposure      = nullptr; // optional 1x1
    const float* reactive      = nullptr; // optional render res [0..1]
    const float* transparencyAndComposition = nullptr;

    int   renderW  = 0, renderH  = 0;
    int   displayW = 0, displayH = 0;

    float jitterOffsetX = 0.0f, jitterOffsetY = 0.0f;
    float motionVectorScaleX = 1.0f, motionVectorScaleY = 1.0f;
    float frameTimeDelta = 16.667f;
    float sharpness = 0.0f;

    bool  reset            = false;
    bool  enableSharpening = false;
    uint32_t flags         = 0;
};
