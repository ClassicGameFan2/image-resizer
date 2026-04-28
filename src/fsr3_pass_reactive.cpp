// =============================================================================
// fsr3_pass_reactive.cpp
// FSR 3.1.5 CPU Port — Passes 7/8: Generate Reactive & TCR Masks
//
// Source: ffx_fsr3upscaler_autogen_reactive_pass.hlsl
//         ffx_fsr3upscaler_tcr_autogenerate_pass.hlsl
//
// Identical algorithm to FSR2 pass 7/8 (fsr2_pass_reactive.cpp).
// For static images: zero reactive mask is used.
// =============================================================================
#include "fsr3_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>
#include <vector>

static inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

void fsr3PassGenerateReactive(
    const float* opaqueColor, const float* compositeColor,
    int w, int h, float* outReactive,
    float cutoffThreshold, float binaryValue, float scale)
{
    for (int i = 0; i < w * h; i++) {
        size_t idx = (size_t)i * 4;
        float oL = luma709(opaqueColor[idx],    opaqueColor[idx+1],    opaqueColor[idx+2]);
        float cL = luma709(compositeColor[idx], compositeColor[idx+1], compositeColor[idx+2]);
        float diff = std::abs(cL - oL);
        outReactive[i] = clamp(
            (diff > cutoffThreshold) ? binaryValue * scale : 0.0f,
            0.0f, 1.0f);
    }
}

void fsr3PassTCRAutogenerate(
    const float* opaqueColor, const float* compositeColor,
    int w, int h, float* outTcr,
    float autoTcrThreshold, float autoTcrScale)
{
    for (int i = 0; i < w * h; i++) {
        size_t idx = (size_t)i * 4;
        float oL   = luma709(opaqueColor[idx],    opaqueColor[idx+1],    opaqueColor[idx+2]);
        float cL   = luma709(compositeColor[idx], compositeColor[idx+1], compositeColor[idx+2]);
        float diff = std::abs(cL - oL);
        outTcr[i]  = clamp(
            saturate(diff / std::max(autoTcrThreshold, 1e-6f)) * autoTcrScale,
            0.0f, 1.0f);
    }
}

void fsr3GenerateZeroReactiveMask(int w, int h, std::vector<float>& out) {
    out.assign((size_t)w * h, 0.0f);
}
