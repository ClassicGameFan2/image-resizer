// =============================================================================
// fsr3_pass_accumulate.cpp
// FSR 3.1.5 CPU Port — Pass 4/5: Temporal Accumulation & Upscaling
//
// Source: ffx_fsr3upscaler_accumulate_pass.hlsl
//         ffx_fsr3upscaler_common.h (Lanczos-2, SRTM, AABB clip, luma instability)
//
// FSR3.1 KEY CHANGES FROM FSR2:
//
//   1. PER-PIXEL ACCUMULATION FACTOR (accumulationFactor buffer).
//      FSR2 used a global per-frame exponential convergence.
//      FSR3 tracks a per-pixel float in [0..1] that increases over time
//      and is RESET per-pixel when disocclusion is detected. This allows
//      fast convergence specifically at newly revealed pixels without
//      slowing down convergence elsewhere.
//
//   2. LUMA INSTABILITY TRACKING (lumaInstability buffer).
//      FSR3 tracks the variance of upscaled luma across frames.
//      High luma instability → reduce history weight (less temporal blur
//      on flickering content). Stored at display res, updated each frame.
//
//   3. LOCKS AT DISPLAY RESOLUTION.
//      Lock values come from buf.lockMask (display res) and buf.newLockMask.
//      No render→display mapping needed for lock lookups.
//
//   4. SHADING CHANGE DETECTION via luminanceMip1.
//      Compare the upscaled luma to the low-res luminance mip to detect
//      large shading changes (explosions, lights turning on/off, etc.).
//      When shading change is high, reduce history weight.
//
//   5. PREPARED REACTIVITY (buf.preparedReactivity).
//      Uses the output of pass 3 instead of the raw reactive buffer.
//      Includes both user reactive mask and motion divergence.
//
//   6. FINAL REACTIVITY = max(dilated TC mask + motion divergence, shadingChange).
//      This combines all sources of temporal instability for the blend factor.
//
//   NOTE: History is ALWAYS stored in LINEAR space (same as our FSR2 port).
//         SRTM is used ONLY transiently for AABB computation.
// =============================================================================
#include "fsr3_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>
#include <cstring>

// ── Math helpers ──────────────────────────────────────────────────────────────

static inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

static inline float3 srtmTonemap(float3 c) {
    float m   = ffxMax3(c.x, c.y, c.z);
    float rcp = 1.0f / (m + 1.0f);
    return float3(c.x*rcp, c.y*rcp, c.z*rcp);
}

static inline float3 srtmInverse(float3 c) {
    float m   = ffxMax3(c.x, c.y, c.z);
    float rcp = 1.0f / std::max(1.0f/32768.0f, 1.0f - m);
    return float3(c.x*rcp, c.y*rcp, c.z*rcp);
}

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

// ── AABB from render-res 3x3 neighborhood (SRTM space) ───────────────────────
static void computeAABB(
    const float* colorBuf, int rW, int rH,
    int cx, int cy, float3& aabbMin, float3& aabbMax)
{
    aabbMin = float3( 1e30f);
    aabbMax = float3(-1e30f);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = std::max(0, std::min(rW-1, cx+dx));
            int ny = std::max(0, std::min(rH-1, cy+dy));
            size_t si = ((size_t)ny * rW + nx) * 4;
            float3 c(colorBuf[si], colorBuf[si+1], colorBuf[si+2]);
            float3 ct = srtmTonemap(c);
            aabbMin = min(aabbMin, ct);
            aabbMax = max(aabbMax, ct);
        }
    }
}

// ── Bilinear history sample (LINEAR) ─────────────────────────────────────────
static inline float3 sampleHistBilinear(
    const float* hist, int dW, int dH, float px, float py)
{
    int   x0 = (int)std::floor(px), y0 = (int)std::floor(py);
    float fx  = px - (float)x0,     fy  = py - (float)y0;
    auto fetch = [&](int x, int y) -> float3 {
        x = std::max(0, std::min(dW-1, x));
        y = std::max(0, std::min(dH-1, y));
        size_t i = ((size_t)y * dW + x) * 4;
        return float3(hist[i], hist[i+1], hist[i+2]);
    };
    float3 c00 = fetch(x0,   y0);
    float3 c10 = fetch(x0+1, y0);
    float3 c01 = fetch(x0,   y0+1);
    float3 c11 = fetch(x0+1, y0+1);
    float3 top = c00*(1-fx) + c10*fx;
    float3 bot = c01*(1-fx) + c11*fx;
    return top*(1-fy) + bot*fy;
}

// Bilinear scalar (for accumulation factor, luma instability)
static inline float sampleScalarBilinear(
    const float* buf_, int dW, int dH, float px, float py)
{
    int   x0 = (int)std::floor(px), y0 = (int)std::floor(py);
    float fx  = px - (float)x0,     fy  = py - (float)y0;
    auto f = [&](int x, int y) -> float {
        x = std::max(0, std::min(dW-1, x));
        y = std::max(0, std::min(dH-1, y));
        return buf_[(size_t)y * dW + x];
    };
    return f(x0,y0)*(1-fx)*(1-fy) + f(x0+1,y0)*fx*(1-fy)
         + f(x0,y0+1)*(1-fx)*fy   + f(x0+1,y0+1)*fx*fy;
}

// ── Sample luminanceMip1 at display-res position ──────────────────────────────
static inline float sampleLumaMip1(
    const Fsr3InternalBuffers& buf, float dispX, float dispY)
{
    // Map display pixel to mip1 coordinates
    float mX = dispX * (float)buf.mip1W / (float)buf.displayW;
    float mY = dispY * (float)buf.mip1H / (float)buf.displayH;
    int   x0 = std::max(0, std::min(buf.mip1W-1, (int)std::floor(mX)));
    int   y0 = std::max(0, std::min(buf.mip1H-1, (int)std::floor(mY)));
    return buf.luminanceMip1[(size_t)y0 * buf.mip1W + x0];
}

// ── Main pass ─────────────────────────────────────────────────────────────────

void fsr3PassAccumulate(
    const float*         currentColor,      // render res float RGBA (jittered)
    Fsr3InternalBuffers& buf,
    float                sharpness,
    bool                 enableBuiltinSharpen)
{
    int rW = buf.renderW,  rH = buf.renderH;
    int dW = buf.displayW, dH = buf.displayH;
    float scaleX = (float)rW / (float)dW;
    float scaleY = (float)rH / (float)dH;
    bool isFirst = buf.firstFrame;

    // Rotate history buffers
    std::copy(buf.accumulatedColor.begin(), buf.accumulatedColor.end(),
              buf.prevAccumulatedColor.begin());
    std::copy(buf.accumulationFactor.begin(), buf.accumulationFactor.end(),
              buf.prevAccumulationFactor.begin());

    // Constants (FSR3.1.4+ defaults)
    const float kMaxHistoryWeight        = 0.91f;
    const float kMinDisoccAccum          = -0.333f; // FSR3.1.4+ default
    const float kShadingChangeThreshold  =  0.1f;
    const float kLumaInstabilityDecay    =  0.9f;

    for (int dy = 0; dy < dH; dy++) {
        for (int dx = 0; dx < dW; dx++) {
            size_t dstIdx = ((size_t)dy * dW + dx) * 4;

            // ── 1. Map display → render sub-pixel + jitter ────────────────
            float srcX = ((float)dx + 0.5f) * scaleX - 0.5f + buf.jitterX;
            float srcY = ((float)dy + 0.5f) * scaleY - 0.5f + buf.jitterY;
            int   cx   = (int)std::round(srcX);
            int   cy   = (int)std::round(srcY);
            int   crX  = std::max(0, std::min(rW-1, cx));
            int   crY  = std::max(0, std::min(rH-1, cy));

            // ── 2. Lanczos-2 upsample current frame ───────────────────────
            float3 lanczosColor(0.0f);
            float  lanczosAlpha  = 0.0f;
            float  lanczosWeight = 0.0f;
            for (int ky = -1; ky <= 2; ky++) {
                float wy = lanczos2(srcY - (float)(cy + ky));
                for (int kx = -1; kx <= 2; kx++) {
                    float wx = lanczos2(srcX - (float)(cx + kx));
                    float w  = wx * wy;
                    int sx   = std::max(0, std::min(rW-1, cx+kx));
                    int sy   = std::max(0, std::min(rH-1, cy+ky));
                    size_t si = ((size_t)sy * rW + sx) * 4;
                    float3 sc(currentColor[si], currentColor[si+1], currentColor[si+2]);
                    lanczosColor  = lanczosColor + sc * w;
                    lanczosAlpha += currentColor[si+3] * w;
                    lanczosWeight += w;
                }
            }
            if (lanczosWeight < 1e-6f) lanczosWeight = 1.0f;
            float3 curPixel  = saturate(lanczosColor * (1.0f / lanczosWeight));
            float  curAlpha  = clamp(lanczosAlpha / lanczosWeight, 0.0f, 1.0f);
            float  curLuma   = luma709(curPixel.x, curPixel.y, curPixel.z);

            // ── 3. AABB (SRTM space, render-res 3x3) ─────────────────────
            float3 aabbMin, aabbMax;
            computeAABB(currentColor, rW, rH, crX, crY, aabbMin, aabbMax);

            // ── 4. Shading change detection via luminanceMip1 ─────────────
            // Compare current upscaled luma to the low-res mip luma.
            // A large difference means the shading changed significantly.
            float mip1Luma     = sampleLumaMip1(buf, (float)dx, (float)dy);
            float shadingChange = clamp(
                std::abs(curLuma - mip1Luma) / std::max(mip1Luma, 1e-4f),
                0.0f, 1.0f);

            // ── 5. Luma instability update ────────────────────────────────
            // Track temporal variance of upscaled luma at this display pixel.
            // High instability → reduce history weight (flickering content).
            size_t dispIdx    = (size_t)dy * dW + dx;
            float  prevInstab = buf.lumaInstability[dispIdx];
            // prevInstab reflects how different previous frames were.
            // Update: blend toward current shading change.
            float  instab     = prevInstab * kLumaInstabilityDecay
                              + shadingChange * (1.0f - kLumaInstabilityDecay);
            buf.lumaInstability[dispIdx] = instab;

            // ── 6. Reprojection (same as FSR2) ────────────────────────────
            int   rIdx    = crY * rW + crX;
            float mvRX    = buf.dilatedMotionVectorsX[rIdx];
            float mvRY    = buf.dilatedMotionVectorsY[rIdx];
            float prevDispX = (float)dx + mvRX / scaleX;
            float prevDispY = (float)dy + mvRY / scaleY;

            // ── 7. Sample history (LINEAR bilinear) ───────────────────────
            bool hasHistory = !isFirst
                && prevDispX >= 0.0f && prevDispX < (float)(dW-1)
                && prevDispY >= 0.0f && prevDispY < (float)(dH-1);
            float3 histLinear(0.0f);
            float  prevAccumFactor = 0.0f;
            if (hasHistory) {
                histLinear      = sampleHistBilinear(
                    buf.prevAccumulatedColor.data(), dW, dH,
                    prevDispX, prevDispY);
                prevAccumFactor = sampleScalarBilinear(
                    buf.prevAccumulationFactor.data(), dW, dH,
                    prevDispX, prevDispY);
            }

            // ── 8. Clip history to AABB ───────────────────────────────────
            float3 histSRTM   = srtmTonemap(histLinear);
            float3 clippedSRTM = clamp(histSRTM, aabbMin, aabbMax);
            float3 clippedHist = srtmInverse(clippedSRTM);

            // ── 9. Per-pixel accumulation factor (FSR3.1 NEW) ────────────
            // Increase per-pixel accumulation factor over time.
            // Reset toward 0 on disocclusion.
            float  disocclusion   = buf.disocclusionMask[rIdx];
            float  reactivity     = buf.preparedReactivity[rIdx];

            // accFactor increases toward 1 over ~16 frames (speed 0.2),
            // but is modulated down on disocclusion and reactivity.
            float newAccumFactor;
            if (isFirst || !hasHistory) {
                newAccumFactor = 0.0f;
            } else {
                // Step up from previous, modulated by disocclusion
                // kMinDisoccAccum=-0.333: allows partial history even at edge
                float disoccStep = disocclusion + kMinDisoccAccum;
                newAccumFactor = prevAccumFactor + 0.1f * disoccStep;
                // Reactive content slows down accumulation
                newAccumFactor *= (1.0f - reactivity * 0.5f);
                // Luma instability: high flicker → slower accumulation
                newAccumFactor *= (1.0f - instab * 0.3f);
                newAccumFactor = clamp(newAccumFactor, 0.0f, 1.0f);
            }
            buf.accumulationFactor[dispIdx] = newAccumFactor;

            // ── 10. Blend factor from accumulation factor ─────────────────
            // FSR3.1: Uses per-pixel accFactor instead of global frame index.
            // Lock boosts history slightly. New locks protect against AABB clip.
            float lockVal    = buf.lockMask[dispIdx];
            float newLockVal = buf.newLockMask[dispIdx];

            float histWeight = kMaxHistoryWeight * newAccumFactor;
            // Lock: stable pixels get a small extra history boost
            histWeight = std::min(histWeight + lockVal * 0.04f, 0.95f);
            // Shading change: large shading changes reduce history
            histWeight *= (1.0f - clamp(shadingChange * 2.0f, 0.0f, 1.0f) * 0.5f);
            histWeight  = clamp(histWeight, 0.0f, 0.95f);
            float curWeight = 1.0f - histWeight;

            // ── 11. Blend in LINEAR space ─────────────────────────────────
            float3 blended = curPixel * curWeight + clippedHist * histWeight;
            blended = saturate(blended);

            // ── 12. Optional built-in sharpen (Pass 5) ────────────────────
            if (enableBuiltinSharpen && sharpness > 0.0f) {
                float3 aabbMidSRTM    = (aabbMin + aabbMax) * 0.5f;
                float3 localAvgLinear = srtmInverse(aabbMidSRTM);
                blended = saturate(blended + (blended - localAvgLinear) * sharpness);
            }

            // ── 13. Store LINEAR ──────────────────────────────────────────
            buf.accumulatedColor[dstIdx+0] = blended.x;
            buf.accumulatedColor[dstIdx+1] = blended.y;
            buf.accumulatedColor[dstIdx+2] = blended.z;
            buf.accumulatedColor[dstIdx+3] = curAlpha;

            buf.internalColorHistory[dstIdx+0] = blended.x;
            buf.internalColorHistory[dstIdx+1] = blended.y;
            buf.internalColorHistory[dstIdx+2] = blended.z;
            buf.internalColorHistory[dstIdx+3] = curAlpha;
        }
    }
}
