// =============================================================================
// fsr2_pass_accumulate.cpp
// FSR 2.3.4 CPU Port — Pass 3/4: Temporal Accumulation & Upscaling
//
// Ported from: ffx_fsr2_accumulate_pass.hlsl
//              ffx_fsr2_common.h (Lanczos-2, SRTM, AABB clip)
//
// KEY DESIGN DECISIONS (faithful to the original FSR2 shader):
//
//   1. HISTORY IS STORED LINEAR. SRTM is used ONLY transiently for AABB
//      computation. Stored and blended values are always linear [0..1].
//      This prevents the darkening-over-frames bug.
//
//   2. JITTER OFFSET IS ADDED (+) when computing srcX/srcY.
//      fsr2ResampleWithShift with shiftX=jX places original pixel p into
//      buffer position p+jX. To recover original position p from the jittered
//      buffer, sample at p+jX (not p-jX).
//
//   3. NO JITTER DELTA IN MOTION VECTOR REPROJECTION.
//      Display-space reprojection is independent of sub-pixel jitter.
//      FSR2 requires jitter-cancelled MVs from the application. Zero MVs
//      are already correct for a static scene.
//
//   4. AABB IS COMPUTED FROM RAW RENDER-RES NEIGHBORS (hard min/max in SRTM
//      space). This matches the original FSR2 shader exactly and is correct
//      for both static images and real 3D scenes. It is a fast 3x3 raw fetch,
//      not a Lanczos upsample — keeping it faithful and performant.
//
//   5. BLEND FACTOR converges toward ~0.91 over ~16 frames, modulated by
//      disocclusion, lock, and reactive masks.
//
//   NOTE ON SHARPNESS WITH JITTER ON:
//      Temporal accumulation without sharpening is deliberately softer than
//      a single-frame upscale. This is correct FSR2 behavior. RCAS (Pass 5)
//      recovers and enhances the detail that accumulation softens. For best
//      results with jitter ON, run with --fsr2-rcas on --fsr2-sharpness 0.2.
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
// Maps linear [0..∞) to [0..1) for perceptually stable min/max clamping.
// Prevents HDR color outliers from over-expanding the AABB bounds.
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
// a=2: 4-tap kernel (radius 2 render pixels each side).
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

// ── AABB from render-res raw neighbors (faithful to FSR2) ─────────────────────
//
// Computes hard min/max of the 3x3 render-res neighborhood centered at
// (cx, cy), in SRTM tonemapped space. This is exactly what the original
// FSR2 accumulate shader does:
//   - Fast: 9 raw texture fetches, no upsampling
//   - Correct for 3D scenes: each render-res pixel represents one world sample
//   - Correct for static images: same neighborhood, same bounds every frame
//     when there is no jitter (jitter-OFF 64-frame case is stable)
//
// The AABB is used to clamp history and prevent ghosting. Values within the
// current frame's local color range are kept; stale out-of-range values are
// clipped toward the nearest in-range color.
static void computeRenderSpaceAABB(
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
            // Tonemap to SRTM space — this value is transient, never stored
            float3 ct = srtmTonemap(c);
            aabbMin = min(aabbMin, ct);
            aabbMax = max(aabbMax, ct);
        }
    }
}

// ── Bilinear fetch from display-res float RGBA history buffer (LINEAR) ─────────
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

    // Ratio: how many render pixels span one display pixel
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
            //   srcX = (dx + 0.5) * scaleX - 0.5
            //
            // Then ADD jitter offset. fsr2ResampleWithShift with shiftX=jX
            // places original pixel p into buffer position p+jX. To sample
            // original world position p from the jittered buffer, read at
            // p+jX (adding the jitter, not subtracting).
            float srcX = ((float)dx + 0.5f) * scaleX - 0.5f + buf.jitterX;
            float srcY = ((float)dy + 0.5f) * scaleY - 0.5f + buf.jitterY;

            int centerRX = (int)std::round(srcX);
            int centerRY = (int)std::round(srcY);
            int clampedRX = std::max(0, std::min(rW - 1, centerRX));
            int clampedRY = std::max(0, std::min(rH - 1, centerRY));

            // ── 2. Lanczos-2 upsample of current jittered frame ───────────
            //
            // Sample a 4x4 render-res neighborhood using Lanczos(x,2) kernel
            // centered at srcX, srcY. This is the core upscaling step,
            // identical to the FSR2 accumulate shader.
            // srcX/srcY includes the jitter offset so each frame hits a
            // different sub-pixel position → temporal super-resolution.
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

            // ── 3. AABB from render-res raw neighbors (faithful to FSR2) ───
            //
            // Hard min/max of the 3x3 render-res neighborhood in SRTM space.
            // This is exactly what the original FSR2 accumulate shader does:
            // fast (9 raw fetches), correct for 3D scenes (each render-res
            // pixel = one world sample), and stable for jitter-OFF static
            // images (history == current → never clipped).
            //
            // For jitter ON: the accumulated history (blend of multiple
            // sub-pixel Lanczos-2 samples) may fall slightly outside this box
            // at edges, causing mild softening. This is expected behavior —
            // RCAS (Pass 5) sharpening is the intended way to recover edge
            // detail after temporal accumulation.
            float3 aabbMin, aabbMax;
            computeRenderSpaceAABB(currentColor, rW, rH,
                                   clampedRX, clampedRY,
                                   aabbMin, aabbMax);

            // ── 4. Reprojection: find display pixel in previous history ────
            //
            // For a static scene: MV = 0, prevDispX = dx (same position).
            // For a 3D app: dilated MVs from pass 1 (jitter-cancelled by
            // the engine per FSR2 spec). No jitter delta is added here —
            // reprojection is in display space, independent of sub-pixel
            // jitter. The jitter only affects srcX/srcY (step 1 above).
            int   rIdx    = clampedRY * rW + clampedRX;
            float mvRX    = buf.dilatedMotionVectorsX[rIdx]; // render-res pixels
            float mvRY    = buf.dilatedMotionVectorsY[rIdx];

            // Convert MV from render space to display space
            float mvDispX = mvRX / scaleX;
            float mvDispY = mvRY / scaleY;

            // Previous display position (MV points current→previous)
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

            // ── 6. Clip history to AABB (in SRTM space) ───────────────────
            //
            // Tonemap history to SRTM space → clamp to render-res AABB →
            // inverse tonemap back to linear. Only the AABB clamping uses
            // SRTM; the stored and blended values remain linear throughout.
            // Valid accumulated colors within the current neighborhood range
            // pass through. Stale ghosted colors are clipped toward the
            // nearest in-range color.
            float3 histSRTM          = srtmTonemap(histColorLinear);
            float3 clippedHistSRTM   = clamp(histSRTM, aabbMin, aabbMax);
            float3 clippedHistLinear = srtmInverse(clippedHistSRTM);

            // ── 7. Compute blend factor ─────────────────────────────────────
            //
            // History weight converges toward kMaxHistoryWeight over ~16
            // frames using an exponential curve. Modulated by:
            //   - disocclusion: newly revealed pixels have no valid history
            //   - lock: stable pixels benefit from slightly more history
            //   - reactive: animated/transparent content uses less history
            float lockVal      = buf.lockMask[rIdx];
            float disocclusion = buf.disocclusionMask[rIdx];
            float reactive     = reactiveBuffer ? reactiveBuffer[rIdx] : 0.0f;

            const float kMaxHistoryWeight = 0.91f;
            const float kConvergenceSpeed = 0.2f;

            float frameConvergence = 0.0f;
            if (!isFirstFrame && buf.frameIndex > 0) {
                frameConvergence = 1.0f - std::exp(
                    -(float)buf.frameIndex * kConvergenceSpeed);
            }

            float historyWeight = kMaxHistoryWeight * frameConvergence;
            historyWeight *= disocclusion;
            historyWeight  = std::min(historyWeight + lockVal * 0.04f, 0.95f);
            historyWeight *= (1.0f - reactive * 0.5f);
            historyWeight  = clamp(historyWeight, 0.0f, 0.95f);
            float currentWeight = 1.0f - historyWeight;

            // ── 8. Blend in LINEAR space ────────────────────────────────────
            //
            // Both currentPixel and clippedHistLinear are linear [0..1].
            // Linear blending produces a linear result with no darkening or
            // nonlinear compounding across frames.
            float3 blended = currentPixel * currentWeight
                           + clippedHistLinear * historyWeight;
            blended = saturate(blended);

            // ── 9. Optional built-in sharpen (Pass 4) ──────────────────────
            //
            // Mild unsharp mask on the blended linear result.
            // For best quality, prefer RCAS (Pass 5) instead.
            if (enableBuiltinSharpen && sharpness > 0.0f) {
                float3 aabbMidSRTM    = (aabbMin + aabbMax) * 0.5f;
                float3 localAvgLinear = srtmInverse(aabbMidSRTM);
                blended = saturate(blended + (blended - localAvgLinear) * sharpness);
            }

            // ── 10. Store results LINEAR ────────────────────────────────────
            //
            // accumulatedColor: becomes prevAccumulatedColor next frame.
            // internalColorHistory: goes to RCAS (pass 5) or direct output.
            // Both store LINEAR values — no tonemap is ever stored.
            buf.accumulatedColor[dstIdx + 0] = blended.x;
            buf.accumulatedColor[dstIdx + 1] = blended.y;
            buf.accumulatedColor[dstIdx + 2] = blended.z;
            buf.accumulatedColor[dstIdx + 3] = currentAlpha;

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
            buf.lockAccum[dIdx] = isFirstFrame
                ? lockVal
                : buf.lockAccum[dIdx] * 0.9f + lockVal * 0.1f;
        }
    }
}
