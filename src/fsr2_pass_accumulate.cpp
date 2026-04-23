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
//      SRTM is used ONLY transiently for computing AABB clamp bounds
//      in a perceptually stable space. The stored and blended values are
//      always linear [0..1]. This prevents darkening-over-frames.
//
//   2. JITTER OFFSET IS ADDED (not subtracted) when computing srcX/srcY.
//      Our jitter frame generator shifts content RIGHT for positive jX
//      (pixel r in jittered buffer = original pixel r-jX). Therefore to
//      recover original source position p from the jittered buffer, sample
//      at p + jX. Subtracting jX would double-shift and cause the image
//      to drift left/down as frames accumulate.
//
//   3. NO JITTER DELTA IN MOTION VECTOR REPROJECTION.
//      Reprojection finds where the current display pixel came from in
//      the display-space history. For a static scene the world point is
//      at the same display position every frame regardless of jitter.
//      The jitter only affects sub-pixel sampling of the current render
//      (srcX/srcY), never the display-space reprojection MV.
//      FSR2 specifies that the application provides jitter-cancelled MVs.
//      Our zero MVs are already correct for a static scene.
//
//   4. AABB COLOR CLAMP (in SRTM space) prevents ghosting without
//      darkening. History is clamped in SRTM space, then inverse-mapped
//      back to linear before blending. Only the AABB computation uses SRTM.
//
//   5. BLEND FACTOR converges toward ~0.91 over ~16 frames, modulated
//      by disocclusion mask, lock mask, and reactive mask.
// =============================================================================
#include "fsr2_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>
#include <cstring>

// ── Math helpers ──────────────────────────────────────────────────────────────

static inline float luma709(float r, float g, float b) {
    // Rec.709 luma — matches FSR2's ffx_fsr2_common.h RGBToLuma exactly
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// SRTM — used ONLY for AABB computation, NEVER for history storage.
// Maps linear [0..∞) to [0..1) for perceptually stable neighborhood clamping.
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

// Lanczos(x, a=2) kernel — matches ffx_fsr2_accumulate_pass.hlsl exactly.
// a=2 gives a 4-tap kernel (radius 2 render pixels each side).
static inline float sincF(float x) {
    if (std::abs(x) < 1e-6f) return 1.0f;
    const float kPi = 3.14159265358979323846f;
    float px = kPi * x;
    return std::sin(px) / px;
}
static inline float lanczos2(float x) {
    x = std::abs(x);
    if (x >= 2.0f) return 0.0f;
    return sincF(x) * sincF(x * 0.5f);
}

// Clamped fetch from render-resolution float RGBA buffer
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

// Bilinear fetch from display-resolution float RGBA history buffer (LINEAR)
static inline float3 sampleHistoryBilinear(const float* hist, int dW, int dH,
                                            float px, float py) {
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
// Purpose: clamp history to prevent ghosting. Done in SRTM space for
// stability — HDR outliers cannot over-expand the AABB this way.
// The AABB is used transiently and never stored.
static void computeSRTM_AABB(
    const float* colorBuf, int rW, int rH,
    int cx, int cy,
    float3& aabbMin, float3& aabbMax)
{
    aabbMin = float3( 1e30f);
    aabbMax = float3(-1e30f);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = std::max(0, std::min(rW - 1, cx + dx));
            int ny = std::max(0, std::min(rH - 1, cy + dy));
            size_t idx = ((size_t)ny * rW + nx) * 4;
            float3 c(colorBuf[idx], colorBuf[idx+1], colorBuf[idx+2]);
            float3 ct = srtmTonemap(c);   // tonemap for AABB only, never stored
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

    // Ratio: how many render pixels per display pixel
    float scaleX = (float)rW / (float)dW;
    float scaleY = (float)rH / (float)dH;

    bool isFirstFrame = buf.firstFrame;

    // Copy current accumulated to previous before overwriting
    std::copy(buf.accumulatedColor.begin(), buf.accumulatedColor.end(),
              buf.prevAccumulatedColor.begin());

    for (int dy = 0; dy < dH; dy++) {
        for (int dx = 0; dx < dW; dx++) {
            size_t dstIdx = ((size_t)dy * dW + dx) * 4;

            // ── 1. Map display pixel to render-res sub-pixel position ──────
            //
            // Standard display→render mapping:
            //   renderPos = (displayPixel + 0.5) * scaleX - 0.5
            //
            // The jitter shifts the content in the render buffer. Our jitter
            // frame generator (fsr2ResampleWithShift with shiftX=jX) places
            // the value originally at render position p into buffer position
            // p + jX (content moves right for positive jX). Therefore to
            // sample the render buffer at the correct position for this display
            // pixel, we ADD the jitter offset:
            //
            //   srcX = renderPos + jitterX
            //
            // NOT subtract. Subtracting would double-shift and cause drift.
            float srcX = ((float)dx + 0.5f) * scaleX - 0.5f + buf.jitterX;
            float srcY = ((float)dy + 0.5f) * scaleY - 0.5f + buf.jitterY;

            // Nearest render pixel to the sub-pixel sample center
            int centerRX = (int)std::round(srcX);
            int centerRY = (int)std::round(srcY);
            int clampedRX = std::max(0, std::min(rW - 1, centerRX));
            int clampedRY = std::max(0, std::min(rH - 1, centerRY));

            // ── 2. Lanczos-2 upsample current jittered frame ───────────────
            //
            // Sample a 4x4 render-res neighborhood using the Lanczos(x,2)
            // kernel centered at srcX, srcY. This is the core upscaling step.
            // Because srcX includes the jitter offset, each frame's Lanczos
            // sample hits a slightly different sub-pixel position, giving
            // temporal super-resolution when accumulated.
            float3 lanczosColor(0.0f);
            float  lanczosAlpha  = 0.0f;
            float  lanczosWeight = 0.0f;

            for (int ky = -1; ky <= 2; ky++) {
                float wy = lanczos2(srcY - (float)(centerRY + ky));
                for (int kx = -1; kx <= 2; kx++) {
                    float wx  = lanczos2(srcX - (float)(centerRX + kx));
                    float w   = wx * wy;
                    int sx    = std::max(0, std::min(rW - 1, centerRX + kx));
                    int sy    = std::max(0, std::min(rH - 1, centerRY + ky));
                    size_t si = ((size_t)sy * rW + sx) * 4;
                    float3 sc(currentColor[si], currentColor[si+1], currentColor[si+2]);
                    float  sa = currentColor[si+3];
                    lanczosColor = lanczosColor + sc * w;
                    lanczosAlpha += sa * w;
                    lanczosWeight += w;
                }
            }

            if (lanczosWeight < 1e-6f) lanczosWeight = 1.0f;
            float3 currentPixel = lanczosColor * (1.0f / lanczosWeight);
            float  currentAlpha = clamp(lanczosAlpha / lanczosWeight, 0.0f, 1.0f);
            // Clamp Lanczos output — Lanczos produces slight ringing at edges
            currentPixel = saturate(currentPixel);

            // ── 3. Compute 3x3 AABB in SRTM space for ghosting prevention ──
            //
            // The AABB is computed from the render-res neighborhood of the
            // current frame (in SRTM space for stability). History is clamped
            // to this box, preventing old stale colors from ghosting through
            // when the scene changes or on disoccluded regions.
            float3 aabbMin, aabbMax;
            computeSRTM_AABB(currentColor, rW, rH, clampedRX, clampedRY,
                              aabbMin, aabbMax);

            // ── 4. Reprojection: find this display pixel in previous history ─
            //
            // We look up where this display pixel came from in the previous
            // frame's display-space history buffer.
            //
            // For a STATIC SCENE: the world point behind display pixel (dx,dy)
            // is at the same display position in every frame, regardless of what
            // jitter the render used. So MV = 0 and prevDispX = dx.
            //
            // For a 3D APP: the dilated motion vectors from pass 1 are used.
            // These are already jitter-cancelled by the 3D engine (FSR2 spec
            // requires applications to provide jitter-cancelled MVs). We use
            // them directly — no jitter delta is added here.
            //
            // This is why we do NOT add a "jitter delta" to the MV. The jitter
            // only affects srcX/srcY (sub-pixel sampling of current render),
            // NOT the display-space reprojection.
            int    rIdx  = clampedRY * rW + clampedRX;
            float  mvRX  = buf.dilatedMotionVectorsX[rIdx]; // render-res pixels
            float  mvRY  = buf.dilatedMotionVectorsY[rIdx];

            // Convert motion vector from render space to display space
            float mvDispX = mvRX / scaleX;
            float mvDispY = mvRY / scaleY;

            // Previous display position (MV points from current→previous)
            float prevDispX = (float)dx + mvDispX;
            float prevDispY = (float)dy + mvDispY;

            // ── 5. Sample history (LINEAR) at reprojected position ─────────
            float3 histColorLinear(0.0f);
            bool hasHistory = !isFirstFrame &&
                              prevDispX >= 0.0f && prevDispX < (float)(dW - 1) &&
                              prevDispY >= 0.0f && prevDispY < (float)(dH - 1);

            if (hasHistory) {
                histColorLinear = sampleHistoryBilinear(
                    buf.prevAccumulatedColor.data(), dW, dH,
                    prevDispX, prevDispY);
            }

            // ── 6. Clip history to AABB (in SRTM space) ────────────────────
            //
            // Tonemap history to SRTM space, clamp to the current frame's
            // AABB, then inverse-tonemap back to linear. This removes ghosting
            // without touching the stored history values (no darkening).
            float3 histSRTM        = srtmTonemap(histColorLinear);
            float3 clippedHistSRTM = clamp(histSRTM, aabbMin, aabbMax);
            float3 clippedHistLinear = srtmInverse(clippedHistSRTM);

            // ── 7. Compute blend factor ─────────────────────────────────────
            //
            // History weight converges toward kMaxHistoryWeight over ~16 frames.
            // Disocclusion kills history (newly revealed pixels have none).
            // Lock adds a small boost (stable pixels benefit from more history).
            // Reactive reduces history (animated/transparent objects change often).
            float lockVal      = buf.lockMask[rIdx];
            float disocclusion = buf.disocclusionMask[rIdx];
            float reactive     = reactiveBuffer ? reactiveBuffer[rIdx] : 0.0f;

            // Convergence: weight = maxWeight * (1 - exp(-frameIndex * speed))
            // At frameIndex=16: 1 - exp(-16*0.2) = 0.96 → clamped to maxWeight
            const float kMaxHistoryWeight = 0.91f;
            const float kConvergenceSpeed = 0.2f;

            float frameConvergence = 0.0f;
            if (!isFirstFrame && buf.frameIndex > 0) {
                frameConvergence = 1.0f - std::exp(
                    -(float)buf.frameIndex * kConvergenceSpeed);
            }

            float historyWeight = kMaxHistoryWeight * frameConvergence;

            // Disocclusion: newly revealed pixels have no valid history
            historyWeight *= disocclusion;

            // Lock: small additive boost for stable pixels, capped at 0.95
            historyWeight = std::min(historyWeight + lockVal * 0.04f, 0.95f);

            // Reactive: transparent/animated content needs less history
            historyWeight *= (1.0f - reactive * 0.5f);

            historyWeight = clamp(historyWeight, 0.0f, 0.95f);
            float currentWeight = 1.0f - historyWeight;

            // ── 8. Blend in LINEAR space ────────────────────────────────────
            //
            // Both currentPixel and clippedHistLinear are linear [0..1].
            // Linear blending produces a linear result — no darkening or
            // nonlinear compounding across frames.
            float3 blended = currentPixel * currentWeight
                           + clippedHistLinear * historyWeight;
            blended = saturate(blended);

            // ── 9. Optional built-in sharpen (Pass 4) ──────────────────────
            //
            // Mild unsharp mask applied to the blended linear result.
            // For best quality, prefer RCAS (Pass 5) instead.
            if (enableBuiltinSharpen && sharpness > 0.0f) {
                // Local average estimated from AABB midpoint, mapped to linear
                float3 aabbMidSRTM    = (aabbMin + aabbMax) * 0.5f;
                float3 localAvgLinear = srtmInverse(aabbMidSRTM);
                float3 sharpened = blended + (blended - localAvgLinear) * sharpness;
                blended = saturate(sharpened);
            }

            // ── 10. Store results ───────────────────────────────────────────
            //
            // accumulatedColor: LINEAR, becomes prevAccumulatedColor next frame
            buf.accumulatedColor[dstIdx + 0] = blended.x;
            buf.accumulatedColor[dstIdx + 1] = blended.y;
            buf.accumulatedColor[dstIdx + 2] = blended.z;
            buf.accumulatedColor[dstIdx + 3] = currentAlpha;

            // internalColorHistory: linear output, ready for RCAS or direct write
            buf.internalColorHistory[dstIdx + 0] = blended.x;
            buf.internalColorHistory[dstIdx + 1] = blended.y;
            buf.internalColorHistory[dstIdx + 2] = blended.z;
            buf.internalColorHistory[dstIdx + 3] = currentAlpha;
        }
    }

    // ── Propagate lock from render-res to display-res ──────────────────────
    for (int dy = 0; dy < dH; dy++) {
        for (int dx = 0; dx < dW; dx++) {
            int    renderX = std::min(rW - 1, (int)((float)dx * scaleX));
            int    renderY = std::min(rH - 1, (int)((float)dy * scaleY));
            float  lockVal = buf.lockMask[(size_t)renderY * rW + renderX];
            size_t dIdx    = (size_t)dy * dW + dx;
            if (isFirstFrame) {
                buf.lockAccum[dIdx] = lockVal;
            } else {
                buf.lockAccum[dIdx] = buf.lockAccum[dIdx] * 0.9f + lockVal * 0.1f;
            }
        }
    }
}
