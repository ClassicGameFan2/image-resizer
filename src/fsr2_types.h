// =============================================================================
// fsr2_types.h
// FSR 2.3.4 CPU Port — Shared types, constants, and internal resource buffers
// Based on: ffx_fsr2.h, ffx_fsr2_common.h, ffx_fsr2_resources.h (AMD MIT)
// =============================================================================
#pragma once
#include "fsr_math.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

// ----------------------------------------------------------------
// FSR2 Context Flags (mirrors FfxFsr2InitializationFlagBits)
// ----------------------------------------------------------------
#define FSR2_ENABLE_DEPTH_INVERTED             (1 << 0)
#define FSR2_ENABLE_DEPTH_INFINITE             (1 << 1)
#define FSR2_ENABLE_AUTO_EXPOSURE              (1 << 2)
#define FSR2_ENABLE_HIGH_DYNAMIC_RANGE         (1 << 3)
#define FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION (1 << 4)

// ----------------------------------------------------------------
// FSR2 Pass IDs (mirrors FfxFsr2Pass)
// ----------------------------------------------------------------
#define FSR2_PASS_DEPTH_CLIP                   0
#define FSR2_PASS_RECONSTRUCT_PREVIOUS_DEPTH   1
#define FSR2_PASS_LOCK                         2
#define FSR2_PASS_ACCUMULATE                   3
#define FSR2_PASS_ACCUMULATE_SHARPEN           4
#define FSR2_PASS_RCAS                         5
#define FSR2_PASS_COMPUTE_LUMINANCE_PYRAMID    6
#define FSR2_PASS_GENERATE_REACTIVE            7
#define FSR2_PASS_TCR_AUTOGENERATE             8
#define FSR2_PASS_COUNT                        9

// ----------------------------------------------------------------
// Internal resource buffers used across passes
// All are float (linear) internally; only input/output use uint8.
// ----------------------------------------------------------------
struct Fsr2InternalBuffers {
    // Render resolution (low-res) sized
    int renderW = 0, renderH = 0;
    // Display resolution (high-res) sized
    int displayW = 0, displayH = 0;

    // Pass 6 output: per-pixel luma at render res (float, 1 channel)
    std::vector<float> luminanceCurrent;   // renderW * renderH
    std::vector<float> luminancePrevious;  // renderW * renderH (shifted from last frame)

    // Auto-exposure: computed from luminance pyramid (1 value)
    float autoExposure = 1.0f;
    float prevAutoExposure = 1.0f;

    // Pass 1 output: dilated motion vectors (float2, render res)
    std::vector<float> dilatedMotionVectorsX; // renderW * renderH
    std::vector<float> dilatedMotionVectorsY;

    // Pass 1 output: dilated depth (float, render res)
    std::vector<float> dilatedDepth;      // renderW * renderH

    // Pass 0 output: disocclusion mask (float, render res)
    std::vector<float> disocclusionMask;  // renderW * renderH  [0=occluded, 1=visible]

    // Pass 2 output: lock mask (float, render res)
    std::vector<float> lockMask;          // renderW * renderH  [0=unlocked, 1=locked]
    // Pass 2: new lock luminance (float, render res)
    std::vector<float> lockLuminance;     // renderW * renderH

    // Pass 3/4 output: accumulated color at display res (float4 RGBA)
    std::vector<float> accumulatedColor;  // displayW * displayH * 4
    // Previous frame accumulated color (for reprojection)
    std::vector<float> prevAccumulatedColor; // displayW * displayH * 4
    // Internal tonemapped history (float4, display res)
    std::vector<float> internalColorHistory; // displayW * displayH * 4

    // Lock accumulation weight (float, display res)
    std::vector<float> lockAccum;         // displayW * displayH

    // Frame counter (for reset detection and luminance history)
    int frameIndex = 0;
    bool firstFrame = true;

    // Jitter offsets used for the current frame (in pixel space)
    float jitterX = 0.0f;
    float jitterY = 0.0f;

    void init(int rW, int rH, int dW, int dH) {
        renderW = rW; renderH = rH;
        displayW = dW; displayH = dH;
        size_t rSize = (size_t)rW * rH;
        size_t dSize = (size_t)dW * dH;

        luminanceCurrent.assign(rSize, 0.0f);
        luminancePrevious.assign(rSize, 0.0f);
        dilatedMotionVectorsX.assign(rSize, 0.0f);
        dilatedMotionVectorsY.assign(rSize, 0.0f);
        dilatedDepth.assign(rSize, 0.5f);
        disocclusionMask.assign(rSize, 1.0f);
        lockMask.assign(rSize, 0.0f);
        lockLuminance.assign(rSize, 0.0f);
        accumulatedColor.assign(dSize * 4, 0.0f);
        prevAccumulatedColor.assign(dSize * 4, 0.0f);
        internalColorHistory.assign(dSize * 4, 0.0f);
        lockAccum.assign(dSize, 0.0f);
        autoExposure = 1.0f;
        prevAutoExposure = 1.0f;
        frameIndex = 0;
        firstFrame = true;
        jitterX = 0.0f; jitterY = 0.0f;
    }

    void reset() {
        std::fill(accumulatedColor.begin(), accumulatedColor.end(), 0.0f);
        std::fill(prevAccumulatedColor.begin(), prevAccumulatedColor.end(), 0.0f);
        std::fill(internalColorHistory.begin(), internalColorHistory.end(), 0.0f);
        std::fill(lockMask.begin(), lockMask.end(), 0.0f);
        std::fill(lockAccum.begin(), lockAccum.end(), 0.0f);
        frameIndex = 0;
        firstFrame = true;
    }
};

// ----------------------------------------------------------------
// FSR2 Dispatch Parameters (mirrors FfxFsr2DispatchDescription)
// All color data as float RGBA [0..1] linear.
// ----------------------------------------------------------------
struct Fsr2DispatchParams {
    // INPUT BUFFERS (render resolution, float RGBA)
    const float* color         = nullptr; // jittered render-res color
    const float* depth         = nullptr; // render-res depth [0..1] (0.5 for flat scenes)
    const float* motionVectors = nullptr; // render-res motion vectors (float2, interleaved XY)
    const float* exposure      = nullptr; // optional 1x1 exposure (nullptr = auto)
    const float* reactive      = nullptr; // optional render-res reactive mask
    const float* transparencyAndComposition = nullptr; // optional T&C mask

    int renderW = 0, renderH = 0;
    int displayW = 0, displayH = 0;

    // Jitter offset (in pixel space, as returned by ffxFsr2GetJitterOffset)
    float jitterOffsetX = 0.0f;
    float jitterOffsetY = 0.0f;

    // Motion vector scale: maps MV values to pixel space
    float motionVectorScaleX = 1.0f;
    float motionVectorScaleY = 1.0f;

    float frameTimeDelta = 16.667f; // milliseconds
    float sharpness = 0.0f;         // 0=off, >0 = sharpen amount

    bool reset = false; // set true on camera cuts
    bool enableSharpening = false;

    uint32_t flags = FSR2_ENABLE_AUTO_EXPOSURE;
};
