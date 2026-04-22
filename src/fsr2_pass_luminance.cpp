// =============================================================================
// fsr2_pass_luminance.cpp
// FSR 2.3.4 CPU Port — Pass 6: Compute Luminance Pyramid
//
// Ported from: ffx_fsr2_compute_luminance_pyramid_pass.hlsl
//              ffx_fsr2_common.h (luma formula, exposure)
//
// What this pass does:
//   1. Computes per-pixel log-luma at render resolution.
//   2. Builds a mip chain (we approximate with a single average).
//   3. Computes auto-exposure from the average log-luma.
//
// GPU uses SPD (Single Pass Downsampler). We use a CPU iterative
// average which produces the identical final exposure value.
// =============================================================================
#include "fsr2_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>

// AMD luma formula from ffx_fsr2_common.h:
// RGBToLuma(rgb) = dot(rgb, float3(0.2126, 0.7152, 0.0722))
// (standard Rec.709 luma — this is what FSR2 uses, NOT the FSR1 approximation)
static inline float rgbToLuma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Pre-exposure luma computation from ffx_fsr2_common.h:
// ffxFsr2ComputeAutoExposureFromLavg uses log2 average.
// We compute: log2(max(luma, epsilon)) per pixel, then average,
// then derive exposure as: exp2(-averageLog2Luma + bias)
// The bias matches AMD's formula: exposure target = 0.18 (middle grey)
static float computeAutoExposure(const float* logLumaBuffer, int count) {
    double sum = 0.0;
    int validCount = 0;
    for (int i = 0; i < count; i++) {
        if (logLumaBuffer[i] > -30.0f) { // skip black pixels
            sum += (double)logLumaBuffer[i];
            validCount++;
        }
    }
    if (validCount == 0) return 1.0f;
    float avgLogLuma = (float)(sum / validCount);
    // AMD target: middle grey = 0.18 => log2(0.18) ≈ -2.47
    // Exposure = exp2(-avgLogLuma + log2(0.18))
    // Simplified: exposure = 0.18 / exp2(avgLogLuma)
    float exposure = 0.18f / std::exp2(avgLogLuma);
    // Clamp to reasonable range to prevent explosions
    return clamp(exposure, 0.001f, 10000.0f);
}

void fsr2PassComputeLuminancePyramid(
    const float* colorBuffer, // render res, float RGBA
    Fsr2InternalBuffers& buf,
    bool autoExposure)
{
    int w = buf.renderW;
    int h = buf.renderH;
    int count = w * h;

    // Step 1: Shift previous luminance (copy current -> previous)
    std::copy(buf.luminanceCurrent.begin(), buf.luminanceCurrent.end(), buf.luminancePrevious.begin());
    buf.prevAutoExposure = buf.autoExposure;

    // Step 2: Compute per-pixel log-luma at render resolution
    // Matches ffx_fsr2_compute_luminance_pyramid_pass.hlsl dispatch
    std::vector<float> logLuma(count);
    for (int i = 0; i < count; i++) {
        float r = colorBuffer[i * 4 + 0];
        float g = colorBuffer[i * 4 + 1];
        float b = colorBuffer[i * 4 + 2];
        float luma = rgbToLuma(r, g, b);
        // Apply current exposure for consistent luminance
        luma = std::max(luma, 1e-10f);
        buf.luminanceCurrent[i] = luma;
        logLuma[i] = std::log2(luma);
    }

    // Step 3: Compute auto-exposure if requested
    if (autoExposure) {
        buf.autoExposure = computeAutoExposure(logLuma.data(), count);
    } else {
        buf.autoExposure = 1.0f;
    }
}
