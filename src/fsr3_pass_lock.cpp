// =============================================================================
// fsr3_pass_lock.cpp
// FSR 3.1.5 CPU Port — Pass 2: Lock Computation (at DISPLAY resolution)
//
// Source: ffx_fsr3upscaler_lock_pass.hlsl
//
// KEY FSR3.1 CHANGE FROM FSR2:
//   Lock is now computed at DISPLAY resolution (not render resolution).
//   This avoids precision issues when comparing lock luminance across frames
//   at render res and saves ALU in the accumulate pass.
//
//   The lock detection logic: compare the Lanczos-upsampled luma of the
//   current frame at this display pixel with its 4 neighbors (also at display
//   res). Low variance → lockable pixel.
//
//   lockLuminance is stored at display res for the accumulate pass to use
//   in shading change detection.
// =============================================================================
#include "fsr3_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>

static inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Lanczos-2 kernel (radius 2, a=2) — same as accumulate pass
static inline float sincF(float x) {
    if (std::abs(x) < 1e-6f) return 1.0f;
    const float kPi = 3.14159265358979323846f;
    float px = kPi * x;
    return std::sin(px) / px;
}
static inline float lanczos2(float x) {
    x = std::abs(x);
    return (x >= 2.0f) ? 0.0f : sincF(x) * sincF(x * 0.5f);
}

// Sample color from render-res jittered buffer via Lanczos-2 upsample.
// This gives us the upscaled luma at the display pixel for lock detection.
static float sampleLumaLanczos(
    const float* colorBuf, int rW, int rH,
    float srcX, float srcY)
{
    int cx = (int)std::round(srcX);
    int cy = (int)std::round(srcY);
    float  weightSum = 0.0f;
    float  lumaSum   = 0.0f;

    for (int ky = -1; ky <= 2; ky++) {
        float wy = lanczos2(srcY - (float)(cy + ky));
        for (int kx = -1; kx <= 2; kx++) {
            float wx = lanczos2(srcX - (float)(cx + kx));
            float w  = wx * wy;
            int sx   = std::max(0, std::min(rW-1, cx+kx));
            int sy   = std::max(0, std::min(rH-1, cy+ky));
            size_t si = ((size_t)sy * rW + sx) * 4;
            float L  = luma709(colorBuf[si], colorBuf[si+1], colorBuf[si+2]);
            lumaSum   += L * w;
            weightSum += w;
        }
    }
    return (weightSum > 1e-6f) ? lumaSum / weightSum : 0.0f;
}

void fsr3PassLock(
    const float*         colorBuffer,  // render res, float RGBA (jittered)
    Fsr3InternalBuffers& buf)
{
    int rW = buf.renderW,  rH = buf.renderH;
    int dW = buf.displayW, dH = buf.displayH;
    float scaleX = (float)rW / (float)dW;
    float scaleY = (float)rH / (float)dH;

    // Build a display-res luma buffer by Lanczos upsampling
    std::vector<float> displayLuma((size_t)dW * dH);
    for (int dy = 0; dy < dH; dy++) {
        for (int dx = 0; dx < dW; dx++) {
            // Same jitter-corrected sampling as accumulate pass
            float srcX = ((float)dx + 0.5f) * scaleX - 0.5f + buf.jitterX;
            float srcY = ((float)dy + 0.5f) * scaleY - 0.5f + buf.jitterY;
            displayLuma[(size_t)dy * dW + dx] =
                sampleLumaLanczos(colorBuffer, rW, rH, srcX, srcY);
        }
    }

    // Lock detection at display res
    auto sL = [&](int x, int y) -> float {
        x = std::max(0, std::min(x, dW-1));
        y = std::max(0, std::min(y, dH-1));
        return displayLuma[(size_t)y * dW + x];
    };

    for (int y = 0; y < dH; y++) {
        for (int x = 0; x < dW; x++) {
            size_t idx = (size_t)y * dW + x;
            float eL = sL(x, y);
            float bL = sL(x,   y-1);
            float dL = sL(x-1, y);
            float fL = sL(x+1, y);
            float hL = sL(x,   y+1);

            // Noise/stability metric (same formula as FSR2 lock pass)
            float nz     = 0.25f * (bL + dL + fL + hL) - eL;
            float maxL   = ffxMax3(ffxMax3(bL, dL, eL), fL, hL);
            float minL   = ffxMin3(ffxMin3(bL, dL, eL), fL, hL);
            float range  = maxL - minL;
            float stability = 1.0f;
            if (range > 1e-6f)
                stability = saturate(1.0f - std::abs(nz) / range);

            // Lock value: [0,1]. Modulated by disocclusion at render res.
            // Map display pixel back to render res for disocclusion fetch.
            int rx = std::min(buf.renderW-1, (int)((float)x * scaleX));
            int ry = std::min(buf.renderH-1, (int)((float)y * scaleY));
            float disocclusion = buf.disocclusionMask[(size_t)ry * buf.renderW + rx];

            float lockVal = (stability > 0.5f ? stability : 0.0f) * disocclusion;
            buf.lockMask[idx]      = lockVal;
            // Store luma at lock time for shading change detection
            buf.lockLuminance[(size_t)ry * buf.renderW + rx] = eL;
        }
    }
}
