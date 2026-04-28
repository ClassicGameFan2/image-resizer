// =============================================================================
// fsr3_pass_luminance.cpp
// FSR 3.1.5 CPU Port — Pass 10: Compute Luminance Pyramid
//
// Source: ffx_fsr3upscaler_compute_luminance_pyramid_pass.hlsl
//         ffx_fsr3upscaler_common.h (luma, SPD, auto-exposure)
//
// FSR3.1 vs FSR2: Adds a downsampled luminance mip (mip1) used by the
// accumulate pass for shading change detection. In the GPU shader, SPD
// produces up to mip6. We approximate with a simple 2x2 box downsample
// to mip1 (1/2 render res), which is all the accumulate pass uses.
// =============================================================================
#include "fsr3_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>

static inline float rgbToLuma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

static float computeAutoExposureFromLogAvg(const std::vector<float>& logLuma) {
    double sum = 0.0;
    int    cnt = 0;
    for (float v : logLuma) {
        if (v > -30.0f) { sum += (double)v; cnt++; }
    }
    if (cnt == 0) return 1.0f;
    float avgLog = (float)(sum / cnt);
    // ffxFsr3UpscalerComputeAutoExposureFromAverageLog:
    // exposure = exp2(targetLuma - avgLog)  where targetLuma = log2(0.18)
    // Equivalent: 0.18 / exp2(avgLog)
    return clamp(0.18f / std::exp2(avgLog), 0.001f, 10000.0f);
}

void fsr3PassComputeLuminancePyramid(
    const float*         colorBuffer,
    Fsr3InternalBuffers& buf,
    bool                 autoExposureEnabled)
{
    int w = buf.renderW, h = buf.renderH;
    int cnt = w * h;

    std::copy(buf.luminanceCurrent.begin(), buf.luminanceCurrent.end(),
              buf.luminancePrevious.begin());
    buf.prevAutoExposure = buf.autoExposure;

    // ── Per-pixel luma and log-luma ───────────────────────────────────────
    std::vector<float> logLuma(cnt);
    for (int i = 0; i < cnt; i++) {
        float r = colorBuffer[i * 4 + 0];
        float g = colorBuffer[i * 4 + 1];
        float b = colorBuffer[i * 4 + 2];
        float L = std::max(rgbToLuma709(r, g, b), 1e-10f);
        buf.luminanceCurrent[i] = L;
        logLuma[i]              = std::log2(L);
    }

    // ── Auto-exposure ─────────────────────────────────────────────────────
    buf.autoExposure = autoExposureEnabled
        ? computeAutoExposureFromLogAvg(logLuma)
        : 1.0f;

    // ── Mip1 (1/2 render res box downsample) ─────────────────────────────
    // Used by accumulate pass to detect shading changes via low-res luma.
    // In the GPU shader this is produced by SPD mip1.
    int mW = buf.mip1W, mH = buf.mip1H;
    for (int my = 0; my < mH; my++) {
        for (int mx = 0; mx < mW; mx++) {
            int sx = mx * 2, sy = my * 2;
            auto sL = [&](int x, int y) -> float {
                x = std::min(x, w - 1);
                y = std::min(y, h - 1);
                return buf.luminanceCurrent[(size_t)y * w + x];
            };
            float L = 0.25f * (sL(sx,sy) + sL(sx+1,sy) + sL(sx,sy+1) + sL(sx+1,sy+1));
            buf.luminanceMip1[(size_t)my * mW + mx] = L;
        }
    }
}
