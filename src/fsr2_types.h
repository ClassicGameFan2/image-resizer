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
// Internal resource buffers used across passes.
// All color data stored as float linear [0..1] internally.
// Only the final input/output uses uint8.
// ----------------------------------------------------------------
struct Fsr2InternalBuffers {
    // Dimensions
    int renderW = 0, renderH = 0;
    int displayW = 0, displayH = 0;

    // Pass 6: per-pixel luma at render res (float, 1 channel)
    std::vector<float> luminanceCurrent;    // renderW * renderH
    std::vector<float> luminancePrevious;   // renderW * renderH

    // Auto-exposure (scalar). For SDR images this stays 1.0.
    // Only used if FSR2_ENABLE_AUTO_EXPOSURE flag is set AND the
    // luminance pyramid pass (pass 6) is enabled.
    float autoExposure = 1.0f;
    float prevAutoExposure = 1.0f;

    // Pass 1: dilated motion vectors (float XY, render res)
    std::vector<float> dilatedMotionVectorsX; // renderW * renderH
    std::vector<float> dilatedMotionVectorsY;

    // Pass 1: dilated depth (float, render res)
    std::vector<float> dilatedDepth;          // renderW * renderH

    // Pass 0: disocclusion mask (float, render res)
    // 1.0 = visible (history valid), 0.0 = disoccluded (history invalid)
    std::vector<float> disocclusionMask;      // renderW * renderH

    // Pass 2: pixel lock mask (float, render res)
    // 1.0 = fully locked (stable), 0.0 = unlocked
    std::vector<float> lockMask;              // renderW * renderH
    std::vector<float> lockLuminance;         // renderW * renderH

    // Pass 3/4: accumulated color at display res (float4 RGBA LINEAR)
    // IMPORTANT: History is stored in LINEAR space, not tonemapped.
    // SRTM is only used transiently inside the accumulate pass for
    // AABB bound computation; it is NOT applied to stored history.
    std::vector<float> accumulatedColor;      // displayW * displayH * 4
    std::vector<float> prevAccumulatedColor;  // displayW * displayH * 4

    // Pass 3/4: final linear output ready for RCAS or direct output
    std::vector<float> internalColorHistory;  // displayW * displayH * 4

    // Lock accumulation at display res (float, for propagating lock from render->display)
    std::vector<float> lockAccum;             // displayW * displayH

    // Frame state
    int frameIndex = 0;
    bool firstFrame = true;

    // Current frame jitter offset (render-res pixel units)
    float jitterX = 0.0f;
    float jitterY = 0.0f;
    // Previous frame jitter offset (render-res pixel units)
    // Used to compute the jitter-delta motion vector for static scenes.
    float prevJitterX = 0.0f;
    float prevJitterY = 0.0f;

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
        prevJitterX = 0.0f; prevJitterY = 0.0f;
    }

    void reset() {
        std::fill(accumulatedColor.begin(), accumulatedColor.end(), 0.0f);
        std::fill(prevAccumulatedColor.begin(), prevAccumulatedColor.end(), 0.0f);
        std::fill(internalColorHistory.begin(), internalColorHistory.end(), 0.0f);
        std::fill(lockMask.begin(), lockMask.end(), 0.0f);
        std::fill(lockAccum.begin(), lockAccum.end(), 0.0f);
        frameIndex = 0;
        firstFrame = true;
        prevJitterX = 0.0f;
        prevJitterY = 0.0f;
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
    const float* motionVectors = nullptr; // render-res motion XY pairs (float, interleaved)
    const float* exposure      = nullptr; // optional 1x1 exposure (nullptr = auto or 1.0)
    const float* reactive      = nullptr; // optional render-res reactive mask [0..1]
    const float* transparencyAndComposition = nullptr;

    int renderW = 0, renderH = 0;
    int displayW = 0, displayH = 0;

    // Jitter offset for the current frame (render-res pixel units, Halton or zero)
    float jitterOffsetX = 0.0f;
    float jitterOffsetY = 0.0f;

    // Motion vector scale: maps provided MV values to pixel units
    float motionVectorScaleX = 1.0f;
    float motionVectorScaleY = 1.0f;

    float frameTimeDelta = 16.667f; // milliseconds (not used in static image mode)
    float sharpness = 0.0f;         // RCAS sharpness in stops

    bool reset = false;             // true on camera cut / first frame
    bool enableSharpening = false;  // true = pass 4 built-in sharpen

    uint32_t flags = 0;             // FSR2_ENABLE_* flags
};
