// =============================================================================
// fsr2_pass_reactive.cpp
// FSR 2.3.4 CPU Port — Passes 7/8: Generate Reactive & T&C Masks (Optional)
//
// Ported from: ffx_fsr2_autogen_reactive_pass.hlsl
//
// What these passes do:
//   Pass 7: Compares an opaque-only color buffer with a combined
//           (opaque + transparent) color buffer to detect alpha-blended
//           objects. Areas where they differ → high reactive value.
//
//   Pass 8: TCR (Transparency & Composition Reactive) mask generation.
//           Uses a similar luminance-based heuristic to detect special
//           rendering regions (raytraced reflections, animated textures, etc.)
//
// Static image adaptation:
//   - If only one color buffer is provided (no separate opaque buffer),
//     we generate a zero reactive mask (fully stable, no transparency).
//   - This is correct: static PNG images have no temporal alpha blending.
//   - Users can provide a custom reactive mask via --fsr2-reactive for
//     artistic control.
//
// For a real 3D app: supply opaqueColor and compositeColor separately.
// =============================================================================
#include "fsr2_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>
#include <vector>

static inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// ----------------------------------------------------------------
// Pass 7: Generate Reactive Mask
// opaqueColor: render res, float RGBA (opaque geometry only)
// compositeColor: render res, float RGBA (opaque + transparent)
// outReactive: render res, float [0..1]
// ----------------------------------------------------------------
void fsr2PassGenerateReactive(
    const float* opaqueColor,
    const float* compositeColor,
    int w, int h,
    float* outReactive,
    float cutoffThreshold,   // default 0.2
    float binaryValue,       // default 0.9  
    float scale)             // default 1.0
{
    for (int i = 0; i < w * h; i++) {
        size_t idx = (size_t)i * 4;
        float oL = luma709(opaqueColor[idx], opaqueColor[idx+1], opaqueColor[idx+2]);
        float cL = luma709(compositeColor[idx], compositeColor[idx+1], compositeColor[idx+2]);

        // Luminance difference indicates transparent/reactive content
        float diff = std::abs(cL - oL);
        float reactive = (diff > cutoffThreshold) ? binaryValue * scale : 0.0f;
        outReactive[i] = clamp(reactive, 0.0f, 1.0f);
    }
}

// ----------------------------------------------------------------
// Pass 8: TCR Autogenerate (Transparency & Composition Reactive)
// This mask signals pixels that need special handling in accumulation.
// ----------------------------------------------------------------
void fsr2PassTCRAutogenerate(
    const float* opaqueColor,
    const float* compositeColor,
    int w, int h,
    float* outTcr,
    float autoTcrThreshold,  // default 0.05
    float autoTcrScale)      // default 1.0
{
    for (int i = 0; i < w * h; i++) {
        size_t idx = (size_t)i * 4;
        float oL = luma709(opaqueColor[idx], opaqueColor[idx+1], opaqueColor[idx+2]);
        float cL = luma709(compositeColor[idx], compositeColor[idx+1], compositeColor[idx+2]);

        // Subtle luminance differences indicate composition artifacts
        float diff = std::abs(cL - oL);
        float tcr = saturate(diff / std::max(autoTcrThreshold, 1e-6f)) * autoTcrScale;
        outTcr[i] = clamp(tcr, 0.0f, 1.0f);
    }
}

// ----------------------------------------------------------------
// Convenience: Generate zero reactive mask (static images)
// ----------------------------------------------------------------
void fsr2GenerateZeroReactiveMask(int w, int h, std::vector<float>& outReactive) {
    outReactive.assign((size_t)w * h, 0.0f);
}
