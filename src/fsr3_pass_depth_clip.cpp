// =============================================================================
// fsr3_pass_depth_clip.cpp
// FSR 3.1.5 CPU Port — Pass 0: Depth Clip (Disocclusion Detection)
//
// Source: ffx_fsr3upscaler_depth_clip_pass.hlsl
//
// FSR3.1 KEY CHANGE FROM FSR2:
//   Disocclusion now computed from a 2x2 depth comparison instead of a
//   single-pixel comparison. This gives a smoother and more accurate
//   disocclusion mask, reducing ghosting on newly revealed pixels.
//
//   The 2x2 comparison: sample the reconstructed previous depth at
//   (prevX±0.5, prevY±0.5) (i.e. the 4 corners of the reprojected pixel),
//   take the min of those 4 samples, then compare against current depth.
//
// fMinDisocclusionAccumulation: default -0.333 (vs +0.333 in FSR2/FSR3.0)
//   Negative value allows slight "favor history" at the disocclusion boundary,
//   which reduces popping and ghosting in newly revealed areas.
// =============================================================================
#include "fsr3_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>

// FSR3.1.4+ default — reduces disocclusion ghosting
static constexpr float kMinDisocclusionAccumulation = -0.333f;

void fsr3PassDepthClip(
    const float*         currentDepth,
    Fsr3InternalBuffers& buf)
{
    int w = buf.renderW, h = buf.renderH;

    auto sampleDilatedDepth = [&](float fx, float fy) -> float {
        // Bilinear sample of dilated depth buffer
        int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
        float tx = fx - (float)x0, ty = fy - (float)y0;
        auto sd = [&](int x, int y) -> float {
            x = std::max(0, std::min(w-1, x));
            y = std::max(0, std::min(h-1, y));
            return buf.dilatedDepth[(size_t)y * w + x];
        };
        return sd(x0,y0)*(1-tx)*(1-ty) + sd(x0+1,y0)*tx*(1-ty)
             + sd(x0,y0+1)*(1-tx)*ty   + sd(x0+1,y0+1)*tx*ty;
    };

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int   idx = y * w + x;
            float mvX = buf.dilatedMotionVectorsX[idx];
            float mvY = buf.dilatedMotionVectorsY[idx];

            float prevX = (float)x - mvX;
            float prevY = (float)y - mvY;

            bool inBounds = (prevX >= 0.0f && prevX < (float)(w-1) &&
                             prevY >= 0.0f && prevY < (float)(h-1));
            if (!inBounds) {
                buf.disocclusionMask[idx] = 0.0f;
                continue;
            }

            // FSR3.1: 2x2 depth comparison at corners of the reprojected pixel.
            // Use the MINIMUM of the 4 corners — most conservative estimate
            // of what the previous frame showed.
            float d00 = sampleDilatedDepth(prevX - 0.5f, prevY - 0.5f);
            float d10 = sampleDilatedDepth(prevX + 0.5f, prevY - 0.5f);
            float d01 = sampleDilatedDepth(prevX - 0.5f, prevY + 0.5f);
            float d11 = sampleDilatedDepth(prevX + 0.5f, prevY + 0.5f);
            float prevDepth = std::min(std::min(d00, d10), std::min(d01, d11));

            float currDepth = currentDepth[idx];

            // Compare: if current depth is much shallower than previous,
            // this is a newly revealed (disoccluded) pixel.
            const float kScale = 4.0f;
            float threshold  = prevDepth / kScale;
            float depthDiff  = prevDepth - currDepth;

            // Smooth step from the FSR3.1 accumulate pass:
            // occlusion ∈ [0,1], 0=disoccluded, 1=valid history
            float occlusion = (depthDiff > threshold) ? 0.0f : 1.0f;

            // 1.0 = occluded (history valid), 0.0 = disoccluded (history invalid)
            // kMinDisocclusionAccumulation is applied in the accumulate pass,
            // not here. The depth clip pass stores the raw binary result.
            buf.disocclusionMask[idx] = occlusion;
        }
    }
}
