// =============================================================================
// fsr2_pass_accumulate.cpp
// FSR 2.3.4 CPU Port — Pass 3/4: Temporal Accumulation & Upscaling
//
// Ported from: ffx_fsr2_accumulate_pass.hlsl
//              ffx_fsr2_common.h (Lanczos-2, SRTM, AABB clip)
//
// KEY DESIGN DECISIONS (matching the actual FSR2 shader behavior):
//
//   1. HISTORY IS STORED LINEAR. We do not tonemap into history.
//      SRTM is used ONLY for computing the AABB clamp bounds in a
//      perceptually stable space. The stored and blended values are
//      always linear [0..1]. This prevents the "darkening over frames"
//      bug caused by nonlinear compounding across the SRTM round-trip.
//
//   2. JITTER DELTA IN REPROJECTION. For a static scene with changing
//      jitter, the reprojection must account for how the jitter changed
//      between the current and previous frame. The motion vector for
//      every pixel is: MV = -(currentJitter - previousJitter).
//      If real motion vectors are provided (3D app), they already encode
//      this and the jitter delta should be added on top.
//      This is the fix for the "blur with jitter on" bug.
//
//   3. LANCZOS-2 UPSAMPLING from render-res to display-res, with the
//      jitter offset subtracted to correctly align the sub-pixel grid.
//      This is the "upscaling" step — identical to the GPU shader.
//
//   4. AABB COLOR CLAMP prevents ghosting. Computed in SRTM space for
//      stability, then the history is clamped in SRTM space before
//      blending, and the result is inverse-tonemapped back to linear.
//      Only the AABB computation touches SRTM — history and output do not.
//
//   5. BLEND FACTOR: history weight converges toward ~0.9 over ~16 frames.
//      Disocclusion, lock, and reactive all modulate this weight.
// =============================================================================
#include "fsr2_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>
#include <cstring>

// ── Math helpers ─────────────────────────────────────────────────────────────

static inline float luma709(float r, float g, float b) {
    // Rec.709 luma — matches FSR2's ffx_fsr2_common.h RGBToLuma
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// SRTM — used ONLY for AABB computation, never for history storage.
// Maps linear [0..1+] to [0..1] for stable neighborhood clamping.
static inline float3 srtmTonemap(float3 c) {
    float m = ffxMax3(c.x, c.y, c.z);
    float rcp = 1.0f / (m + 1.0f);
    return float3(c.x * rcp, c.y * rcp, c.z * rcp);
}

static inline float3 srtmInverse(float3 c) {
    float m = ffxMax3(c.x, c.y, c.z);
    float rcp = 1.0f / std::max(1.0f / 32768.0f, 1.0f - m);
    return float3(c.x * rcp, c.y * rcp, c.z * rcp);
}

// Lanczos(x, a=2) kernel — matches ffx_fsr2_accumulate_pass.hlsl
static inline float sincF(float x) {
    if (std::abs(x) < 1e-6f) return 1.0f;
    float px = 3.14159265358979323846f * x;
    return std::sin(px) / px;
}
static inline float lanczos2(float x) {
    x = std::abs(x);
    if (x >= 2.0f) return 0.0f;
    return sincF(x) * sincF(x * 0.5f);
}

// Safe fetch from render-res float RGBA buffer
static inline float3 fetchRender(const float* buf, int w, int h, int x, int y) {
    x = std::max(0, std::min(x, w - 1));
    y = std::max(0, std::min(y, h - 1));
    size_t idx = ((size_t)y * w + x) * 4;
    return float3(buf[idx], buf[idx+1], buf[idx+2]);
}
static inline float fetchRenderAlpha(const float* buf, int w, int h, int x, int y) {
    x = std::max(0, std::min(x, w - 1));
    y = std::max(0, std::min(y, h - 1));
    return buf[((size_t)y * w + x) * 4 + 3];
}

// Bilinear fetch from display-res float RGBA history buffer (LINEAR)
static inline float3 sampleHistoryBilinear(const float* hist, int dW, int dH, float px, float py) {
    int x0 = (int)std::floor(px);
    int y0 = (int)std::floor(py);
    float fx = px - (float)x0;
    float fy = py - (float)y0;

    auto fetch = [&](int x, int y) -> float3 {
        x = std::max(0, std::min(dW - 1, x));
        y = std::max(0, std::min(dH - 1, y));
        size_t i = ((size_t)y * dW + x) * 4;
        return float3(hist[i], hist[i+1], hist[i+2]);
    };

    float3 c00 = fetch(x0,   y0);
    float3 c10 = fetch(x0+1, y0);
    float3 c01 = fetch(x0,   y0+1);
    float3 c11 = fetch(x0+1, y0+1);

    float3 top = c00 * (1.0f - fx) + c10 * fx;
    float3 bot = c01 * (1.0f - fx) + c11 * fx;
    return top * (1.0f - fy) + bot * fy;
}

// Compute 3x3 AABB of current render pixels in SRTM space.
// The AABB is used to clamp the history to prevent ghosting.
// We work in SRTM space here for perceptual stability (prevents
// HDR outliers from over-expanding the AABB).
static void computeSRTM_AABB(
    const float* colorBuf, int rW, int rH,
    int cx, int cy,
    float3& aabbMin, float3& aabbMax)
{
    aabbMin = float3(1e30f);
    aabbMax = float3(-1e30f);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = std::max(0, std::min(rW - 1, cx + dx));
            int ny = std::max(0, std::min(rH - 1, cy + dy));
            size_t idx = ((size_t)ny * rW + nx) * 4;
            float3 c(colorBuf[idx], colorBuf[idx+1], colorBuf[idx+2]);
            // Tonemap for AABB only — this value is never stored
            float3 ct = srtmTonemap(c);
            aabbMin = min(aabbMin, ct);
            aabbMax = max(aabbMax, ct);
        }
    }
}

// ── Main pass ─────────────────────────────────────────────────────────────────

void fsr2PassAccumulate(
    const float* currentColor,      // render res, float RGBA (jittered)
    const float* reactiveBuffer,    // render res, float [0..1] or nullptr
    Fsr2InternalBuffers& buf,
    float sharpness,
    bool enableBuiltinSharpen)
{
    int rW = buf.renderW, rH = buf.renderH;
    int dW = buf.displayW, dH = buf.displayH;

    float scaleX = (float)rW / (float)dW;
    float scaleY = (float)rH / (float)dH;

    bool isFirstFrame = buf.firstFrame;

    // Copy current accumulated to previous before we overwrite it
    std::copy(buf.accumulatedColor.begin(), buf.accumulatedColor.end(),
              buf.prevAccumulatedColor.begin());

    // ── Compute jitter delta motion vector ──────────────────────────────────
    // For a static scene with jitter, the motion between frames is purely
    // caused by the jitter offset changing. Every pixel has the same MV:
    //   MV_render = -(currentJitter - previousJitter)
    // This points from the current frame's position back to where the same
    // world point was in the previous frame.
    //
    // If real motion vectors are provided by a 3D app (via dilatedMotionVectors),
    // they are used instead. The jitter delta is already baked into real MVs
    // because the 3D app provides jitter-cancelled motion vectors.
    //
    // For our static image case, dilatedMotionVectors are zero (set by
    // fsr2_context.cpp when pass 1 is disabled, or by pass 1 for a flat scene).
    // We add the jitter delta on top.
    float jitterDeltaX = -(buf.jitterX - buf.prevJitterX); // render-res pixels
    float jitterDeltaY = -(buf.jitterY - buf.prevJitterY);

    for (int dy = 0; dy < dH; dy++) {
        for (int dx = 0; dx < dW; dx++) {
            size_t dstIdx = ((size_t)dy * dW + dx) * 4;

            // ── 1. Map display pixel to render-res sub-pixel position ──────
            // The jitter offset shifts the render image by (jitterX, jitterY)
            // pixels. We subtract it here to "undo" the jitter and find the
            // canonical render-space position for this display pixel.
            float srcX = ((float)dx + 0.5f) * scaleX - 0.5f - buf.jitterX;
            float srcY = ((float)dy + 0.5f) * scaleY - 0.5f - buf.jitterY;

            int centerRX = (int)std::round(srcX);
            int centerRY = (int)std::round(srcY);
            int clampedRX = std::max(0, std::min(rW - 1, centerRX));
            int clampedRY = std::max(0, std::min(rH - 1, centerRY));

            // ── 2. Lanczos-2 upsample current jittered frame ───────────────
            // Sample a 4x4 neighborhood centered at srcX, srcY.
            // This is the core upscaling step — identical to the FSR2 shader.
            float3 lanczosColor(0.0f);
            float  lanczosAlpha = 0.0f;
            float  lanczosWeight = 0.0f;

            for (int ky = -1; ky <= 2; ky++) {
                float wy = lanczos2(srcY - (float)(centerRY + ky));
                for (int kx = -1; kx <= 2; kx++) {
                    float wx = lanczos2(srcX - (float)(centerRX + kx));
                    float w = wx * wy;
                    int sx = std::max(0, std::min(rW - 1, centerRX + kx));
                    int sy = std::max(0, std::min(rH - 1, centerRY + ky));
                    size_t sIdx = ((size_t)sy * rW + sx) * 4;
                    float3 sc(currentColor[sIdx], currentColor[sIdx+1], currentColor[sIdx+2]);
                    float  sa = currentColor[sIdx+3];
                    lanczosColor = lanczosColor + sc * w;
                    lanczosAlpha += sa * w;
                    lanczosWeight += w;
                }
            }

            if (lanczosWeight < 1e-6f) lanczosWeight = 1.0f;
            float3 currentPixel = lanczosColor * (1.0f / lanczosWeight);
            float  currentAlpha = clamp(lanczosAlpha / lanczosWeight, 0.0f, 1.0f);
            // Clamp Lanczos output — Lanczos can produce slight over/undershoot
            currentPixel = saturate(currentPixel);

            // ── 3. Compute AABB in SRTM space for ghosting prevention ──────
            // This is used to clamp history color, preventing old colors from
            // dominating the blend when the scene changes (or between jitter frames).
            float3 aabbMin, aabbMax;
            computeSRTM_AABB(currentColor, rW, rH, clampedRX, clampedRY, aabbMin, aabbMax);

            // ── 4. Reprojection: find this pixel's location in previous frame ──
            // Get the dilated motion vector at this render-res location.
            // This is either from pass 1 (real MV from 3D app) or zero (static).
            int rIdx = clampedRY * rW + clampedRX;
            float mvX = buf.dilatedMotionVectorsX[rIdx]; // render-res pixels
            float mvY = buf.dilatedMotionVectorsY[rIdx];

            // Add jitter delta so the reprojection lands correctly when
            // the jitter offset changed between frames.
            mvX += jitterDeltaX;
            mvY += jitterDeltaY;

            // Convert motion vector from render-res to display-res pixels
            float mvDispX = mvX / scaleX;
            float mvDispY = mvY / scaleY;

            // Previous position in display space
            float prevDispX = (float)dx + mvDispX;
            float prevDispY = (float)dy + mvDispY;

            // ── 5. Sample history at reprojected position ──────────────────
            // History is stored LINEAR. No tonemap on the stored values.
            float3 histColorLinear(0.0f);
            bool hasHistory = !isFirstFrame &&
                              prevDispX >= 0.0f && prevDispX < (float)(dW - 1) &&
                              prevDispY >= 0.0f && prevDispY < (float)(dH - 1);

            if (hasHistory) {
                histColorLinear = sampleHistoryBilinear(
                    buf.prevAccumulatedColor.data(), dW, dH, prevDispX, prevDispY);
            }

            // ── 6. Clip history to AABB (in SRTM space) ────────────────────
            // Tonemap history into SRTM space, clamp to AABB, then invert.
            // This prevents ghosting without darkening the accumulated result.
            float3 histSRTM = srtmTonemap(histColorLinear);
            float3 clippedHistSRTM = clamp(histSRTM, aabbMin, aabbMax);
            float3 clippedHistLinear = srtmInverse(clippedHistSRTM);

            // ── 7. Compute blend factor ─────────────────────────────────────
            float lockVal      = buf.lockMask[rIdx];
            float disocclusion = buf.disocclusionMask[rIdx];
            float reactive     = reactiveBuffer ? reactiveBuffer[rIdx] : 0.0f;

            // History weight converges to kMaxHistoryWeight over ~16 frames.
            // Formula: weight = maxWeight * (1 - exp(-frameIndex * speed))
            // At frame 16: 1 - exp(-16*0.2) ≈ 0.96 → clamped to 0.91
            const float kMaxHistoryWeight  = 0.91f;
            const float kConvergenceSpeed  = 0.2f;

            float frameConvergence = 0.0f;
            if (!isFirstFrame && buf.frameIndex > 0) {
                frameConvergence = 1.0f - std::exp(
                    -(float)buf.frameIndex * kConvergenceSpeed);
            }

            float historyWeight = kMaxHistoryWeight * frameConvergence;

            // Disocclusion kills history (newly revealed pixels have no valid history)
            historyWeight *= disocclusion;

            // Lock: stable pixels keep more history (protects fine detail)
            // Small additive boost capped so we never exceed 0.95
            historyWeight = std::min(historyWeight + lockVal * 0.04f, 0.95f);

            // Reactive: animated/transparent pixels use less history
            historyWeight *= (1.0f - reactive * 0.5f);

            historyWeight = clamp(historyWeight, 0.0f, 0.95f);
            float currentWeight = 1.0f - historyWeight;

            // ── 8. Blend in LINEAR space ────────────────────────────────────
            // Both currentPixel and clippedHistLinear are linear [0..1].
            // Blending linear values gives a linear result — no darkening.
            float3 blended = currentPixel * currentWeight + clippedHistLinear * historyWeight;
            blended = saturate(blended);

            // ── 9. Optional built-in sharpen (Pass 4) ──────────────────────
            // Applied as a mild unsharp mask on the blended linear result.
            // For best quality, use RCAS (Pass 5) instead.
            if (enableBuiltinSharpen && sharpness > 0.0f) {
                // Unsharp mask: sharpen = center + strength * (center - local_avg)
                // We approximate local_avg using the AABB midpoint in linear space
                float3 aabbMidSRTM = (aabbMin + aabbMax) * 0.5f;
                float3 localAvgLinear = srtmInverse(aabbMidSRTM);
                float3 sharpened = blended + (blended - localAvgLinear) * sharpness;
                blended = saturate(sharpened);
            }

            // ── 10. Store results ───────────────────────────────────────────
            // accumulatedColor: LINEAR, used as history next frame
            buf.accumulatedColor[dstIdx + 0] = blended.x;
            buf.accumulatedColor[dstIdx + 1] = blended.y;
            buf.accumulatedColor[dstIdx + 2] = blended.z;
            buf.accumulatedColor[dstIdx + 3] = currentAlpha;

            // internalColorHistory: linear output for RCAS or direct write
            buf.internalColorHistory[dstIdx + 0] = blended.x;
            buf.internalColorHistory[dstIdx + 1] = blended.y;
            buf.internalColorHistory[dstIdx + 2] = blended.z;
            buf.internalColorHistory[dstIdx + 3] = currentAlpha;
        }
    }

    // ── Propagate lock from render-res to display-res ──────────────────────
    for (int dy = 0; dy < dH; dy++) {
        for (int dx = 0; dx < dW; dx++) {
            int renderX = std::min(rW - 1, (int)((float)dx * scaleX));
            int renderY = std::min(rH - 1, (int)((float)dy * scaleY));
            float lockVal = buf.lockMask[(size_t)renderY * rW + renderX];
            size_t dIdx = (size_t)dy * dW + dx;
            if (isFirstFrame) {
                buf.lockAccum[dIdx] = lockVal;
            } else {
                buf.lockAccum[dIdx] = buf.lockAccum[dIdx] * 0.9f + lockVal * 0.1f;
            }
        }
    }
}
