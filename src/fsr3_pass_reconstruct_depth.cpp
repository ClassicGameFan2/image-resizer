// =============================================================================
// fsr3_pass_reconstruct_depth.cpp
// FSR 3.1.5 CPU Port — Pass 1: Reconstruct Previous Depth & Dilate MVs
//
// Source: ffx_fsr3upscaler_reconstruct_dilated_velocity_and_previous_depth_pass.hlsl
//
// Functionally identical to the FSR2 pass. Dilates MVs using 3x3 max-depth
// neighbor search. For static images: zero MVs, uniform depth — a no-op.
// =============================================================================
#include "fsr3_types.h"
#include <cmath>
#include <algorithm>

void fsr3PassReconstructPreviousDepth(
    const float*         depthBuffer,
    const float*         motionVectorBuffer,
    Fsr3InternalBuffers& buf)
{
    int w = buf.renderW, h = buf.renderH;

    auto getDepth = [&](int x, int y) -> float {
        x = std::max(0, std::min(x, w-1));
        y = std::max(0, std::min(y, h-1));
        return depthBuffer[y * w + x];
    };
    auto getMV = [&](int x, int y, float& mvX, float& mvY) {
        x = std::max(0, std::min(x, w-1));
        y = std::max(0, std::min(y, h-1));
        int idx = (y * w + x) * 2;
        mvX = motionVectorBuffer[idx];
        mvY = motionVectorBuffer[idx + 1];
    };

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float bestDepth = -1.0f, bestMvX = 0.0f, bestMvY = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    float d = getDepth(x+dx, y+dy);
                    if (d > bestDepth) {
                        bestDepth = d;
                        getMV(x+dx, y+dy, bestMvX, bestMvY);
                    }
                }
            }
            int idx = y * w + x;
            buf.dilatedDepth[idx]            = bestDepth;
            buf.dilatedMotionVectorsX[idx]   = bestMvX;
            buf.dilatedMotionVectorsY[idx]   = bestMvY;
        }
    }
}
