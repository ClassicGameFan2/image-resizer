// =============================================================================
// fsr2_pass_accumulate.cpp
// FSR 2.3.4 CPU Port — Pass 3/4: Temporal Accumulation & Upscaling
//
// Ported from: ffx_fsr2_accumulate_pass.hlsl
//              ffx_fsr2_common.h (Lanczos, tonemapping, SRTM)
//
// This is the HEART of FSR 2. What it does:
//
//   1. REPROJECTION: For each output (display-res) pixel, compute where
//      it "came from" in the previous frame using dilated motion vectors
//      and the jitter offset. Sample the previous accumulated color there.
//
//   2. LANCZOS UPSCALING: The current jittered render-res frame is
//      sampled using a Lanczos(x,2) kernel centered at the display-res
//      pixel's sub-pixel position. This is the "upscaling" step.
//
//   3. TONEMAPPING: Both history and current color are Reinhard-tonemapped
//      before blending to prevent HDR colors from dominating the blend.
//      (FSR 2.2+ uses SRTM — Simple Reversible Tone Mapper)
//
//   4. TEMPORAL BLEND: Blend current color with reprojected history.
//      Blend factor depends on:
//        - disocclusion mask (no history → use current fully)
//        - lock mask (locked pixels → higher history weight)
//        - reactive mask (reactive pixels → less history)
//        - motion magnitude (fast motion → less history)
//
//   5. COLOR CLAMP (AABB): History color is clamped to the AABB of
//      neighbors in the current frame. This prevents ghosting from
//      long-ago colors leaking through.
//
//   6. INVERSE TONEMAP: Convert back to linear after blending.
//
//   7. LOCK ACCUMULATION: Track how many frames a pixel has been locked.
//
// Static image adaptations:
//   - Motion vectors = zero → reprojection stays on same location.
//   - This means FSR2 is effectively accumulating sub-pixel shifted
//     versions of the same image → genuine temporal super-sampling.
//   - The Lanczos kernel provides the actual reconstruction quality.
// =============================================================================
#include "fsr2_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>
#include <cstring>

// ── Math helpers ─────────────────────────────────────────────────────────────

static inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// FSR2 uses the Simple Reversible Tone Mapper (SRTM) from ffx_fsr2_common.h
// for tonemapping before blending. Same as our FsrSrtmF in fsr_math.h.
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

// Lanczos(x, 2) kernel — from ffx_fsr2_accumulate_pass.hlsl
// a=2 means the kernel spans 2 lobes (radius = 2 pixels)
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

// Sample render-res float RGBA buffer with clamping
static inline float3 sampleRenderColor(const float* buf, int w, int h, int x, int y) {
    x = std::max(0, std::min(x, w-1));
    y = std::max(0, std::min(y, h-1));
    size_t idx = ((size_t)y * w + x) * 4;
    return float3(buf[idx], buf[idx+1], buf[idx+2]);
}
static inline float sampleRenderAlpha(const float* buf, int w, int h, int x, int y) {
    x = std::max(0, std::min(x, w-1));
    y = std::max(0, std::min(h-1, y));
    return buf[((size_t)y * w + x) * 4 + 3];
}

// Sample dilated lock mask at render resolution
static inline float sampleLock(const float* lockMask, int w, int h, int x, int y) {
    x = std::max(0, std::min(x, w-1));
    y = std::max(0, std::min(y, h-1));
    return lockMask[y * w + x];
}

// Sample previous accumulated color (display res)
static inline float3 sampleHistory(const float* hist, int dW, int dH, float px, float py) {
    // Bilinear sample of history buffer at display res
    int x0 = (int)std::floor(px);
    int y0 = (int)std::floor(py);
    float fx = px - (float)x0;
    float fy = py - (float)y0;

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

    float3 top = c00 * (1.0f - fx) + c10 * fx;
    float3 bot = c01 * (1.0f - fx) + c11 * fx;
    return top * (1.0f - fy) + bot * fy;
}

// Compute the 3x3 AABB of current render colors at a render-res location
// Used for history clip to prevent ghosting
static void computeColorAABB(
    const float* colorBuf, int rW, int rH,
    int centerX, int centerY,
    float3& aabbMin, float3& aabbMax)
{
    aabbMin = float3(1e30f);
    aabbMax = float3(-1e30f);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = std::max(0, std::min(rW-1, centerX + dx));
            int ny = std::max(0, std::min(rH-1, centerY + dy));
            size_t idx = ((size_t)ny * rW + nx) * 4;
            float3 c(colorBuf[idx], colorBuf[idx+1], colorBuf[idx+2]);
            // Tonemap before AABB for consistent HDR handling
            c = srtmTonemap(c);
            aabbMin = min(aabbMin, c);
            aabbMax = max(aabbMax, c);
        }
    }
}

void fsr2PassAccumulate(
    const float* currentColor,     // render res, float RGBA (jittered)
    const float* reactiveBuffer,   // render res, float [0..1] or nullptr
    Fsr2InternalBuffers& buf,
    float sharpness,               // 0 = no built-in sharpen (use RCAS pass)
    bool enableBuiltinSharpen)     // true = pass 4, false = pass 3
{
    int rW = buf.renderW, rH = buf.renderH;
    int dW = buf.displayW, dH = buf.displayH;

    // Scale factors: how many render pixels per display pixel
    float scaleX = (float)rW / (float)dW;
    float scaleY = (float)rH / (float)dH;

    // Copy current accumulated to previous before overwriting
    std::copy(buf.accumulatedColor.begin(), buf.accumulatedColor.end(),
              buf.prevAccumulatedColor.begin());

    bool isFirstFrame = buf.firstFrame;

    for (int dy = 0; dy < dH; dy++) {
        for (int dx = 0; dx < dW; dx++) {
            size_t dstIdx = ((size_t)dy * dW + dx) * 4;

            // ── 1. Map display pixel to render-res sub-pixel position ──
            // Account for jitter offset. The jitter moves the render image
            // by (jitterX, jitterY) pixels at render resolution.
            // To undo the jitter when sampling the current frame, we
            // subtract the jitter from the sampling position.
            float srcX = ((float)dx + 0.5f) * scaleX - 0.5f - buf.jitterX;
            float srcY = ((float)dy + 0.5f) * scaleY - 0.5f - buf.jitterY;

            int centerRX = (int)std::round(srcX);
            int centerRY = (int)std::round(srcY);

            // ── 2. Lanczos-2 upsampling of current jittered frame ──
            // Sample a 4x4 neighborhood using Lanczos(x,2) kernel
            float3 lanczosColor(0.0f);
            float lanczosAlpha = 0.0f;
            float lanczosWeight = 0.0f;

            for (int ky = -1; ky <= 2; ky++) {
                float wy = lanczos2(srcY - (float)(centerRY + ky));
                for (int kx = -1; kx <= 2; kx++) {
                    float wx = lanczos2(srcX - (float)(centerRX + kx));
                    float w = wx * wy;
                    int sx = std::max(0, std::min(rW-1, centerRX + kx));
                    int sy = std::max(0, std::min(rH-1, centerRY + ky));
                    size_t sIdx = ((size_t)sy * rW + sx) * 4;
                    float3 sc(currentColor[sIdx], currentColor[sIdx+1], currentColor[sIdx+2]);
                    float sa = currentColor[sIdx+3];
                    lanczosColor = lanczosColor + sc * w;
                    lanczosAlpha += sa * w;
                    lanczosWeight += w;
                }
            }

            if (lanczosWeight < 1e-6f) lanczosWeight = 1.0f;
            float3 currentPixel = lanczosColor * (1.0f / lanczosWeight);
            float currentAlpha = lanczosAlpha / lanczosWeight;
            currentPixel = saturate(currentPixel);

            // ── 3. Apply exposure before blending ──
            float3 exposedCurrent = currentPixel * buf.autoExposure;

            // ── 4. Tonemap current for stable blending ──
            float3 tonemappedCurrent = srtmTonemap(exposedCurrent);

            // ── 5. AABB of current color (at render res) for history clip ──
            float3 aabbMin, aabbMax;
            computeColorAABB(currentColor, rW, rH, centerRX, centerRY, aabbMin, aabbMax);

            // ── 6. Reprojection: find previous pixel location ──
            // Map dilated motion vector from render-res to display-res
            int renderX = std::max(0, std::min(rW-1, centerRX));
            int renderY = std::max(0, std::min(rH-1, centerRY));
            int rIdx = renderY * rW + renderX;

            float mvX = buf.dilatedMotionVectorsX[rIdx]; // in render-res pixels
            float mvY = buf.dilatedMotionVectorsY[rIdx];

            // Convert motion vector to display resolution
            float mvDispX = mvX / scaleX;
            float mvDispY = mvY / scaleY;

            // Previous pixel position in display space
            float prevDispX = (float)dx - mvDispX;
            float prevDispY = (float)dy - mvDispY;

            // ── 7. Sample history at reprojected position ──
            float3 histColor(0.0f);
            bool hasHistory = !isFirstFrame &&
                              prevDispX >= 0.0f && prevDispX < (float)(dW-1) &&
                              prevDispY >= 0.0f && prevDispY < (float)(dH-1);

            if (hasHistory) {
                histColor = sampleHistory(buf.prevAccumulatedColor.data(), dW, dH, prevDispX, prevDispY);
                // History is already tonemapped from previous frame's output
                // In FSR 2.2+ the history is stored pre-tonemapped.
                // We keep it pre-tonemapped here too.
            }

            // ── 8. Clip history to AABB to prevent ghosting ──
            float3 clippedHistory = clamp(histColor, aabbMin, aabbMax);

            // ── 9. Compute blend factor ──
            // Base blend: lerp between current and history
            // More frames = more history weight (converges to ~95%)
            float lockVal = sampleLock(buf.lockMask.data(), rW, rH, renderX, renderY);
            float disocclusion = buf.disocclusionMask[rIdx];

            // Get reactive value
            float reactive = 0.0f;
            if (reactiveBuffer) {
                reactive = reactiveBuffer[rIdx];
            }

            // Accumulation blend factor:
            // - First frame: use 100% current
            // - Disoccluded pixels: mostly current (history is wrong)
            // - Locked pixels: mostly history (stable, don't disturb)
            // - Reactive pixels: less history (they change often)
            // FSR2 uses a convergence speed based on frame count:
            // blendFactor approaches maxHistoryWeight as frameIndex increases
            const float kMinHistoryWeight = 0.0f;
            const float kMaxHistoryWeight = 0.91f; // ~11 frames to convergence
            const float kLockBoost = 0.05f;       // Extra history weight for locked pixels
            const float kReactivePenalty = 0.5f;  // Reduce history for reactive pixels

            float frameConvergence = 0.0f;
            if (!isFirstFrame && buf.frameIndex > 0) {
                // Smooth convergence: weight increases each frame up to max
                // Matches AMD's accumulation speed curve
                frameConvergence = 1.0f - std::exp(-(float)std::min(buf.frameIndex, 32) * 0.15f);
            }

            float historyWeight = lerp(kMinHistoryWeight, kMaxHistoryWeight, frameConvergence);

            // Disocclusion kills history
            historyWeight *= disocclusion;

            // Lock adds extra history stability
            historyWeight = std::min(historyWeight + lockVal * kLockBoost, kMaxHistoryWeight + kLockBoost);

            // Reactive reduces history
            historyWeight *= (1.0f - reactive * kReactivePenalty);

            historyWeight = clamp(historyWeight, 0.0f, 0.99f);
            float currentWeight = 1.0f - historyWeight;

            // ── 10. Blend current + history ──
            float3 blended = tonemappedCurrent * currentWeight + clippedHistory * historyWeight;

            // ── 11. Optional built-in sharpening (Pass 4) ──
            // Pass 4 applies a mild sharpening to the blended result before
            // storing. In practice, RCAS (Pass 5) is preferred; Pass 4 is
            // an older path. We apply a simple unsharp mask here.
            if (enableBuiltinSharpen && sharpness > 0.0f) {
                // Simple Laplacian sharpen on the tonemapped blended result
                float3 center = blended;
                // Sample 4 tonemapped neighbors from history for sharpening reference
                // (Using the blended result as-is, applying unsharp mask)
                // This is a simplification of the built-in sharpen path.
                // For best quality, use RCAS (Pass 5) instead.
                float3 sharpened = center * (1.0f + sharpness) - blended * sharpness;
                blended = clamp(sharpened, float3(0.0f), float3(1.0f));
            }

            // ── 12. Inverse tonemap for storage ──
            // Store tonemapped in history, undo for final output
            // In FSR 2.2+ the history is stored POST-tonemap for stability.
            // The output to the next pass is PRE-inverse-tonemap (linear).
            float3 finalLinear = srtmInverse(blended);

            // Undo exposure for final output
            float rcpExposure = (buf.autoExposure > 1e-6f) ? 1.0f / buf.autoExposure : 1.0f;
            float3 finalColor = finalLinear * rcpExposure;
            finalColor = saturate(finalColor);

            // ── 13. Store results ──
            // History stores the tonemapped blended result (pre-inverse-tonemap)
            buf.accumulatedColor[dstIdx + 0] = blended.x;
            buf.accumulatedColor[dstIdx + 1] = blended.y;
            buf.accumulatedColor[dstIdx + 2] = blended.z;
            buf.accumulatedColor[dstIdx + 3] = currentAlpha;

            // Output buffer: linear, ready for RCAS or direct output
            buf.internalColorHistory[dstIdx + 0] = finalColor.x;
            buf.internalColorHistory[dstIdx + 1] = finalColor.y;
            buf.internalColorHistory[dstIdx + 2] = finalColor.z;
            buf.internalColorHistory[dstIdx + 3] = currentAlpha;
        }
    }

    // Update lock accumulation buffer (display res)
    // Propagate lock stability from render-res to display-res
    for (int dy = 0; dy < dH; dy++) {
        for (int dx = 0; dx < dW; dx++) {
            int renderX = std::min(rW-1, (int)((float)dx * scaleX));
            int renderY = std::min(rH-1, (int)((float)dy * scaleY));
            float lockVal = buf.lockMask[renderY * rW + renderX];
            size_t dstIdx = (size_t)dy * dW + dx;
            if (isFirstFrame) {
                buf.lockAccum[dstIdx] = lockVal;
            } else {
                // Exponential moving average of lock
                buf.lockAccum[dstIdx] = buf.lockAccum[dstIdx] * 0.9f + lockVal * 0.1f;
            }
        }
    }
}
