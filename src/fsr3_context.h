// =============================================================================
// fsr3_context.h
// FSR 3.1.5 CPU Port — Main context and orchestration header
// =============================================================================
#pragma once
#include "fsr3_types.h"
#include "fsr2_jitter.h"
#include "fsr2_jitter_resample.h"
#include <vector>
#include <set>
#include <string>
#include <cstdint>

// ── Pass function forward declarations ────────────────────────────────────────
void fsr3PassComputeLuminancePyramid(
    const float* colorBuffer, Fsr3InternalBuffers& buf, bool autoExposure);

void fsr3PassReconstructPreviousDepth(
    const float* depthBuffer, const float* motionVectorBuffer,
    Fsr3InternalBuffers& buf);

void fsr3PassDepthClip(
    const float* currentDepth, Fsr3InternalBuffers& buf);

void fsr3PassPrepareReactivity(
    const float* reactiveBuffer, const float* tcBuffer,
    Fsr3InternalBuffers& buf);

void fsr3PassLock(
    const float* colorBuffer, Fsr3InternalBuffers& buf);

void fsr3PassNewLocks(
    const float* colorBuffer, Fsr3InternalBuffers& buf);

void fsr3PassAccumulate(
    const float* currentColor, Fsr3InternalBuffers& buf,
    float sharpness, bool enableBuiltinSharpen);

void fsr3PassRCAS(
    const float* input, int w, int h, float* output,
    float sharpnessInStops, bool useDenoise);

void fsr3PassGenerateReactive(
    const float* opaqueColor, const float* compositeColor,
    int w, int h, float* outReactive,
    float cutoffThreshold, float binaryValue, float scale);

void fsr3PassTCRAutogenerate(
    const float* opaqueColor, const float* compositeColor,
    int w, int h, float* outTcr,
    float autoTcrThreshold, float autoTcrScale);

void fsr3GenerateZeroReactiveMask(int w, int h, std::vector<float>& out);

// ── Context ───────────────────────────────────────────────────────────────────
struct Fsr3Context {
    Fsr3InternalBuffers buf;
    bool initialized = false;

    void init(int renderW, int renderH, int displayW, int displayH) {
        buf.init(renderW, renderH, displayW, displayH);
        initialized = true;
    }
    void reset() { buf.reset(); }
};

// ── Dispatch ──────────────────────────────────────────────────────────────────
void fsr3Dispatch(
    Fsr3Context& ctx,
    const Fsr3DispatchParams& params,
    const std::set<int>& enabledPasses,
    float* outDisplayColor,
    bool useRcasDenoise = false);
