// =============================================================================
// fsr2_pass_lock.cpp
// FSR 2.3.4 CPU Port — Pass 2: Pixel Lock Computation
//
// Ported from: ffx_fsr2_lock_pass.hlsl
//
// What this pass does:
//   1. Detects "lockable" pixels — stable, non-noisy pixels that FSR2
//      can safely accumulate from many frames (e.g. flat solid colors).
//   2. Locks protect stable pixels from being overwritten by ghosting
//      correction. They act as anchors for the accumulation.
//   3. Lock detection is based on luma instability: compare current
//      pixel's luma to its ring of 4 neighbors. Low variance → lock.
//
// Key FSR 2.2+ change: lock luminance is stored separately.
// lowerLimiterMultiplier was added to protect against negative RCAS output.
//
// Static image behavior:
//   - All pixels have consistent luma (no temporal noise between jitter frames).
//   - Most non-edge pixels will be flagged as "lockable".
//   - Edge pixels (high luma variance) will not be locked.
//   - This is CORRECT and GOOD for static images.
// =============================================================================
#include "fsr2_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>

// Rec.709 luma (matches FSR2 ffx_fsr2_common.h)
static inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

void fsr2PassLock(
    const float* colorBuffer,    // render res, float RGBA
    Fsr2InternalBuffers& buf)
{
    int w = buf.renderW;
    int h = buf.renderH;

    auto sampleLuma = [&](int x, int y) -> float {
        x = std::max(0, std::min(x, w - 1));
        y = std::max(0, std::min(y, h - 1));
        int idx = (y * w + x) * 4;
        return luma709(colorBuffer[idx], colorBuffer[idx+1], colorBuffer[idx+2]);
    };

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;

            float eL = sampleLuma(x,   y);
            float bL = sampleLuma(x,   y-1);
            float dL = sampleLuma(x-1, y);
            float fL = sampleLuma(x+1, y);
            float hL = sampleLuma(x,   y+1);

            // Noise level = avg of ring - center (from ffx_fsr2_lock_pass.hlsl)
            float nz = 0.25f * (bL + dL + fL + hL) - eL;
            float maxL = ffxMax3(ffxMax3(bL, dL, eL), fL, hL);
            float minL = ffxMin3(ffxMin3(bL, dL, eL), fL, hL);
            float range = maxL - minL;

            // Normalized noise: 0=noisy, 1=stable
            float stabilityFactor = 1.0f;
            if (range > 1e-6f) {
                stabilityFactor = saturate(1.0f - std::abs(nz) / range);
            }

            // Lock threshold: if stability > 0.5, pixel is "lockable"
            // In FSR 2.2+ this also uses luma instability from previous frames.
            // For our CPU port (single pass per frame), we use the stability directly.
            // Lock value: 1.0 = fully locked, 0.0 = unlocked
            float lockVal = (stabilityFactor > 0.5f) ? stabilityFactor : 0.0f;

            // Apply disocclusion: disoccluded pixels cannot be locked
            lockVal *= buf.disocclusionMask[idx];

            buf.lockMask[idx] = lockVal;
            buf.lockLuminance[idx] = eL;
        }
    }
}
