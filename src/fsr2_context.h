// =============================================================================
// fsr2_context.h
// FSR 2.3.4 CPU Port — Main context and orchestration header
// =============================================================================
#pragma once
#include "fsr2_types.h"
#include "fsr2_jitter.h"
#include "fsr2_jitter_resample.h"
#include <vector>
#include <set>
#include <string>
#include <cstdint>

// Forward declarations of pass functions
void fsr2PassComputeLuminancePyramid(const float* colorBuffer, Fsr2InternalBuffers& buf, bool autoExposure);
void fsr2PassReconstructPreviousDepth(const float* depthBuffer, const float* motionVectorBuffer, Fsr2InternalBuffers& buf);
void fsr2PassDepthClip(const float* currentDepth, Fsr2InternalBuffers& buf);
void fsr2PassLock(const float* colorBuffer, Fsr2InternalBuffers& buf);
void fsr2PassAccumulate(const float* currentColor, const float* reactiveBuffer, Fsr2InternalBuffers& buf, float sharpness, bool enableBuiltinSharpen);
void fsr2PassRCAS(const float* input, int w, int h, float* output, float sharpnessInStops, bool useDenoise);
void fsr2PassGenerateReactive(const float* opaqueColor, const float* compositeColor, int w, int h, float* outReactive, float cutoffThreshold, float binaryValue, float scale);
void fsr2PassTCRAutogenerate(const float* opaqueColor, const float* compositeColor, int w, int h, float* outTcr, float autoTcrThreshold, float autoTcrScale);
void fsr2GenerateZeroReactiveMask(int w, int h, std::vector<float>& outReactive);

// ----------------------------------------------------------------
// Fsr2Context: Manages internal state for one upscaling session.
// One context = one (renderW,renderH) → (displayW,displayH) upscale.
// ----------------------------------------------------------------
struct Fsr2Context {
    Fsr2InternalBuffers buf;
    bool initialized = false;

    void init(int renderW, int renderH, int displayW, int displayH) {
        buf.init(renderW, renderH, displayW, displayH);
        initialized = true;
    }

    void reset() {
        buf.reset();
    }
};

// ----------------------------------------------------------------
// fsr2Dispatch: Run one frame of FSR2 upscaling.
//
// The enabledPasses set controls which passes run.
// Default: all 9 passes (0..8).
// Use --onlyenablepasses to restrict to a subset.
//
// Returns: output in outDisplayColor (display res, float RGBA [0..1])
// ----------------------------------------------------------------
void fsr2Dispatch(
    Fsr2Context& ctx,
    const Fsr2DispatchParams& params,
    const std::set<int>& enabledPasses,
    float* outDisplayColor,          // display res output, float RGBA
    bool useRcasDenoise = false);
