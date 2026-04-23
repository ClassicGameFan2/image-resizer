// =============================================================================
// fsr2_pass_luminance.cpp
// FSR 2.3.4 CPU Port — Pass 6: Compute Luminance Pyramid
//
// Ported from: ffx_fsr2_compute_luminance_pyramid_pass.hlsl
//              ffx_fsr2_common.h (luma formula, auto-exposure)
//
// What this pass does:
//   1. Computes per-pixel Rec.709 luma at render resolution.
//   2. Computes a log-average luma (approximates the GPU SPD mip chain).
//   3. Derives an auto-exposure value from the log-average.
//
// IMPORTANT: For SDR images (pixel values already in [0..1]), the
// auto-exposure formula typically produces values < 1.0 (because
// average image luma > 0.18 middle grey). We compute it but store
// it for reference. The accumulate pass does NOT multiply the image
// by autoExposure — that would cause the darkening bug. AutoExposure
// is meaningful in HDR pipelines where scene luminance is in [0..FP16_MAX].
// For our SDR static-image use case, autoExposure is informational only.
//
// GPU uses SPD (Single Pass Downsampler). We use a CPU average which
// produces the same final exposure value as the full mip chain.
// =============================================================================
#include "fsr2_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>

// Rec.709 luma — matches FSR2's ffx_fsr2_common.h exactly
static inline float rgbToLuma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Compute auto-exposure from log-average luma.
// Formula from ffx_fsr2_common.h ffxFsr2ComputeAutoExposureFromLavg:
//   exposure = 0.18 / exp2(avgLogLuma)
// This targets middle grey at 0.18. For SDR images (average luma ~0.3),
// this gives autoExposure ~0.6. We store it but do not apply it in the
// accumulate pass to avoid SDR darkening.
static float computeAutoExposureFromLogAvg(const std::vector<float>& logLuma) {
    double sum = 0.0;
    int count = 0;
    for (float v : logLuma) {
        if (v > -30.0f) { // skip near-black pixels
            sum += (double)v;
            count++;
        }
    }
    if (count == 0) return 1.0f;
    float avgLogLuma = (float)(sum / count);
    float exposure = 0.18f / std::exp2(avgLogLuma);
    // Clamp to sane HDR range. For SDR this is typically 0.3-0.7.
    return clamp(exposure, 0.001f, 10000.0f);
}

void fsr2PassComputeLuminancePyramid(
    const float* colorBuffer,  // render res, float RGBA
    Fsr2InternalBuffers& buf,
    bool autoExposureEnabled)
{
    int w = buf.renderW;
    int h = buf.renderH;
    int count = w * h;

    // Shift previous luma (current becomes previous for lock pass)
    std::copy(buf.luminanceCurrent.begin(), buf.luminanceCurrent.end(),
              buf.luminancePrevious.begin());
    buf.prevAutoExposure = buf.autoExposure;

    // Compute per-pixel luma and log-luma
    std::vector<float> logLuma(count);
    for (int i = 0; i < count; i++) {
        float r = colorBuffer[i * 4 + 0];
        float g = colorBuffer[i * 4 + 1];
        float b = colorBuffer[i * 4 + 2];
        float luma = rgbToLuma709(r, g, b);
        luma = std::max(luma, 1e-10f);
        buf.luminanceCurrent[i] = luma;
        logLuma[i] = std::log2(luma);
    }

    // Compute auto-exposure (stored for reference / HDR use, not applied to SDR)
    if (autoExposureEnabled) {
        buf.autoExposure = computeAutoExposureFromLogAvg(logLuma);
    } else {
        buf.autoExposure = 1.0f;
    }
}
