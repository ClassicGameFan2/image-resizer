// =============================================================================
// fsr2_pass_rcas.cpp
// FSR 2.3.4 CPU Port — Pass 5: RCAS Sharpening on FSR2 output
//
// Ported from: ffx_fsr2_rcas_pass.hlsl
//
// IMPORTANT FSR 2.2+ change vs FSR 1.2.2:
//   In FSR 2.2+, RCAS does NOT apply tonemapping — it operates directly
//   on the linear upscaled output (unlike FSR 1.x which ran RCAS on
//   tonemapped data). This gives better HDR range preservation.
//
// This is essentially the same RCAS algorithm as in fsr_rcas.cpp but
// adapted to operate on float buffers (not uint8) and without the
// FSR1-specific lowerLimiterMultiplier (which is in FSR1's RCAS).
//
// FSR2 RCAS sharpness is passed as a pre-computed constant (a linear
// multiplier in [0..1]), whereas FSR1 RCAS uses stops (exp2(-sharpness)).
// We expose both formats via the sharpnessInStops parameter.
// =============================================================================
#include "fsr2_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>

#define FSR2_RCAS_LIMIT (0.25f - (1.0f / 16.0f))

static inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

static inline float3 sampleF(const float* buf, int w, int h, int x, int y) {
    x = std::max(0, std::min(x, w-1));
    y = std::max(0, std::min(y, h-1));
    size_t idx = ((size_t)y * w + x) * 4;
    return float3(buf[idx], buf[idx+1], buf[idx+2]);
}
static inline float sampleAlphaF(const float* buf, int w, int h, int x, int y) {
    x = std::max(0, std::min(x, w-1));
    y = std::max(0, std::min(y, h-1));
    return buf[((size_t)y * w + x) * 4 + 3];
}

// Apply RCAS to a float RGBA buffer (display resolution).
// Input and output are float [0..1] linear.
void fsr2PassRCAS(
    const float* input,    // display res, float RGBA linear
    int w, int h,
    float* output,         // display res, float RGBA
    float sharpnessInStops,// converted to linear: exp2(-sharpness)
    bool useDenoise)
{
    float sharpConfig = std::exp2(-sharpnessInStops);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float3 b  = sampleF(input, w, h, x,   y-1);
            float3 d  = sampleF(input, w, h, x-1, y  );
            float3 e  = sampleF(input, w, h, x,   y  );
            float3 f  = sampleF(input, w, h, x+1, y  );
            float3 h_ = sampleF(input, w, h, x,   y+1);

            float bL = luma709(b.x, b.y, b.z);
            float dL = luma709(d.x, d.y, d.z);
            float eL = luma709(e.x, e.y, e.z);
            float fL = luma709(f.x, f.y, f.z);
            float hL = luma709(h_.x, h_.y, h_.z);

            // Noise detection (same as FSR1 RCAS)
            float nz = 0.25f * bL + 0.25f * dL + 0.25f * fL + 0.25f * hL - eL;
            float maxL = ffxMax3(ffxMax3(bL, dL, eL), fL, hL);
            float minL = ffxMin3(ffxMin3(bL, dL, eL), fL, hL);
            nz = saturate(std::abs(nz) / std::max(maxL - minL, 1e-6f));
            nz = -0.5f * nz + 1.0f;

            float3 mn4 = min(min(min(b, d), f), h_);
            float3 mx4 = max(max(max(b, d), f), h_);

            // FSR 2.3.4 RCAS: No lowerLimiterMultiplier (that was FSR1's fix).
            // FSR2 RCAS operates on linear data that is already well-conditioned.
            float3 hitMin = float3(
                mn4.x / std::max(4.0f * mx4.x, 1e-6f),
                mn4.y / std::max(4.0f * mx4.y, 1e-6f),
                mn4.z / std::max(4.0f * mx4.z, 1e-6f)
            );
            float3 hitMax = float3(
                (1.0f - mx4.x) / std::min(4.0f * mn4.x - 4.0f, -1e-6f),
                (1.0f - mx4.y) / std::min(4.0f * mn4.y - 4.0f, -1e-6f),
                (1.0f - mx4.z) / std::min(4.0f * mn4.z - 4.0f, -1e-6f)
            );

            float3 lobeRGB = max(-hitMin, hitMax);
            float lobe = std::max(
                -FSR2_RCAS_LIMIT,
                std::min(ffxMax3(lobeRGB.x, lobeRGB.y, lobeRGB.z), 0.0f)
            ) * sharpConfig;

            if (useDenoise) lobe *= nz;

            float rcpL = 1.0f / (4.0f * lobe + 1.0f);
            float3 finalColor = (b * lobe + d * lobe + h_ * lobe + f * lobe + e) * rcpL;
            finalColor = clamp(finalColor, float3(0.0f), float3(1.0f));

            float alpha = sampleAlphaF(input, w, h, x, y);
            size_t dstIdx = ((size_t)y * w + x) * 4;
            output[dstIdx + 0] = finalColor.x;
            output[dstIdx + 1] = finalColor.y;
            output[dstIdx + 2] = finalColor.z;
            output[dstIdx + 3] = alpha;
        }
    }
}
