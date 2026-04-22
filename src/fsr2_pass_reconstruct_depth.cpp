// =============================================================================
// fsr2_pass_reconstruct_depth.cpp
// FSR 2.3.4 CPU Port — Pass 1: Reconstruct Previous Depth & Dilate Motion Vectors
//
// Ported from: ffx_fsr2_reconstruct_previous_depth_pass.hlsl
//
// What this pass does (at render resolution):
//   1. Dilates motion vectors using a 3x3 max-depth neighbor search.
//      This ensures moving objects' motion vectors "bleed" into
//      surrounding pixels (critical for correct reprojection).
//   2. Reconstructs the dilated depth for the previous frame by
//      applying motion vectors to the current depth buffer.
//
// Static image adaptations:
//   - Motion vectors = zero (static scene, zero motion everywhere).
//   - Depth = uniform flat depth (user-specified or 0.5 default).
//   - Dilation still runs (no-op on zero MVs, but keeps code path intact).
// =============================================================================
#include "fsr2_types.h"
#include <cmath>
#include <algorithm>

void fsr2PassReconstructPreviousDepth(
    const float* depthBuffer,         // render res, float [0..1]
    const float* motionVectorBuffer,  // render res, float XY pairs [in pixels]
    Fsr2InternalBuffers& buf)
{
    int w = buf.renderW;
    int h = buf.renderH;

    auto getDepth = [&](int x, int y) -> float {
        x = std::max(0, std::min(x, w - 1));
        y = std::max(0, std::min(y, h - 1));
        return depthBuffer[y * w + x];
    };

    auto getMV = [&](int x, int y, float& mvX, float& mvY) {
        x = std::max(0, std::min(x, w - 1));
        y = std::max(0, std::min(y, h - 1));
        int idx = (y * w + x) * 2;
        mvX = motionVectorBuffer[idx + 0];
        mvY = motionVectorBuffer[idx + 1];
    };

    // For each render-res pixel, find the neighbor with maximum depth
    // (closest to camera in standard depth convention) in a 3x3 window.
    // This is the "dilation" step from the shader.
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float bestDepth = -1.0f;
            float bestMvX = 0.0f, bestMvY = 0.0f;
            int bestDepthIdx = y * w + x;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    float d = getDepth(nx, ny);
                    // FSR2 uses max depth for dilation (finds "nearest" object)
                    // Convention: higher value = closer in non-inverted depth.
                    // For our static scene with uniform depth, all equal.
                    if (d > bestDepth) {
                        bestDepth = d;
                        float mvx, mvy;
                        getMV(nx, ny, mvx, mvy);
                        bestMvX = mvx;
                        bestMvY = mvy;
                    }
                }
            }

            int idx = y * w + x;
            buf.dilatedDepth[idx] = bestDepth;
            buf.dilatedMotionVectorsX[idx] = bestMvX;
            buf.dilatedMotionVectorsY[idx] = bestMvY;
        }
    }
}
