// =============================================================================
// fsr2_pass_depth_clip.cpp
// FSR 2.3.4 CPU Port — Pass 0: Depth Clip (Disocclusion Detection)
//
// Ported from: ffx_fsr2_depth_clip_pass.hlsl
//
// What this pass does:
//   Generates a disocclusion mask. For each render-res pixel:
//   1. Reprojects using dilated motion vectors to find where this
//      pixel "was" in the previous frame.
//   2. Compares reprojected depth to the reconstructed previous depth.
//   3. If the difference is large, marks as disoccluded (new pixel).
//
// Static image adaptation:
//   - Zero motion vectors → reprojection lands on same pixel.
//   - Uniform depth → depth difference = 0 → no disocclusions.
//   - Result: disocclusionMask = 1.0 everywhere (all pixels are "visible").
//   - This is CORRECT behavior for a static scene.
//
// For a real 3D app: supply real depth and motion vectors.
// =============================================================================
#include "fsr2_types.h"
#include <cmath>
#include <algorithm>

void fsr2PassDepthClip(
    const float* currentDepth,       // render res, float [0..1]
    Fsr2InternalBuffers& buf)
{
    int w = buf.renderW;
    int h = buf.renderH;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;

            // Compute reprojected position using dilated motion vectors
            float mvX = buf.dilatedMotionVectorsX[idx];
            float mvY = buf.dilatedMotionVectorsY[idx];

            // Reprojected UV in render resolution
            float prevX = (float)x - mvX;
            float prevY = (float)y - mvY;

            // Check if reprojected position is within bounds
            bool inBounds = (prevX >= 0.0f && prevX < (float)(w - 1) &&
                             prevY >= 0.0f && prevY < (float)(h - 1));

            if (!inBounds) {
                // Out of bounds → disoccluded (new pixel, no history)
                buf.disocclusionMask[idx] = 0.0f;
                continue;
            }

            // Bilinearly sample the dilated depth at the reprojected position
            int px = (int)std::floor(prevX);
            int py = (int)std::floor(prevY);
            float fx = prevX - (float)px;
            float fy = prevY - (float)py;

            auto sampleDepth = [&](int sx, int sy) -> float {
                sx = std::max(0, std::min(w - 1, sx));
                sy = std::max(0, std::min(h - 1, sy));
                return buf.dilatedDepth[sy * w + sx];
            };

            float prevDepth =
                sampleDepth(px,   py)   * (1.0f - fx) * (1.0f - fy) +
                sampleDepth(px+1, py)   * fx           * (1.0f - fy) +
                sampleDepth(px,   py+1) * (1.0f - fx) * fy           +
                sampleDepth(px+1, py+1) * fx           * fy;

            float currDepth = currentDepth[idx];

            // Depth clip threshold: from ffx_fsr2_depth_clip_pass.hlsl
            // A pixel is disoccluded if its depth is significantly LESS than
            // what was reconstructed in the previous frame at that location.
            // (In non-inverted depth: smaller value = further away → newly revealed)
            // Threshold: if currDepth < prevDepth * threshold → disoccluded
            const float kDepthClipBaseScale = 4.0f;
            float threshold = prevDepth / kDepthClipBaseScale;
            float depthDiff = prevDepth - currDepth;

            // 1.0 = occluded (history is valid), 0.0 = disoccluded (history invalid)
            // We use a smooth step so the transition isn't harsh
            float occlusion = (depthDiff > threshold) ? 0.0f : 1.0f;
            buf.disocclusionMask[idx] = occlusion;
        }
    }
}
