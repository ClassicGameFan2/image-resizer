// =============================================================================
// fsr3_pass_prepare_reactivity.cpp
// FSR 3.1.5 CPU Port — Pass 3: Prepare Reactivity
//
// Source: ffx_fsr3upscaler_prepare_reactivity.h (used in accumulate pass)
//
// NEW PASS IN FSR3.1 (does not exist in FSR2):
//   Combines two sources of reactivity into a single buffer:
//     1. Dilated T&C (Transparency & Composition) mask: reflects user-supplied
//        transparent/special rendering regions at render res, dilated using
//        the same 3x3 max approach as motion vectors.
//     2. Motion divergence: detects areas where motion vectors differ
//        significantly from their neighbors — these are object boundaries,
//        fast-moving edges, or newly occluded areas.
//   Final reactivity = max(dilatedTcMask, motionDivergence)
//
// For static images: no TC mask, no motion → preparedReactivity = 0 everywhere.
// =============================================================================
#include "fsr3_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>

void fsr3PassPrepareReactivity(
    const float*         reactiveBuffer,  // render res [0..1], can be nullptr
    const float*         tcBuffer,        // render res [0..1], can be nullptr
    Fsr3InternalBuffers& buf)
{
    int w = buf.renderW, h = buf.renderH;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;

            // ── Dilated TC mask ───────────────────────────────────────────
            // Dilate the TC mask with 3x3 max (same pattern as dilated MVs).
            // This bleeds TC regions into their surroundings.
            float maxTc = 0.0f;
            if (tcBuffer) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = std::max(0, std::min(w-1, x+dx));
                        int ny = std::max(0, std::min(h-1, y+dy));
                        maxTc = std::max(maxTc, tcBuffer[(size_t)ny * w + nx]);
                    }
                }
            }

            // ── Motion divergence ─────────────────────────────────────────
            // Measure how much the motion vector at (x,y) differs from its
            // 4 cardinal neighbors. High divergence = edge/boundary of motion.
            float mvX = buf.dilatedMotionVectorsX[idx];
            float mvY = buf.dilatedMotionVectorsY[idx];

            float divergence = 0.0f;
            const int offsets[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
            for (auto& o : offsets) {
                int nx = std::max(0, std::min(w-1, x+o[0]));
                int ny = std::max(0, std::min(h-1, y+o[1]));
                int nIdx = ny * w + nx;
                float nmX = buf.dilatedMotionVectorsX[nIdx];
                float nmY = buf.dilatedMotionVectorsY[nIdx];
                float dX  = mvX - nmX, dY = mvY - nmY;
                divergence = std::max(divergence,
                    std::sqrt(dX*dX + dY*dY) * 0.25f);
            }
            // Normalize divergence: 1 pixel/frame motion difference → reactive=1
            divergence = clamp(divergence, 0.0f, 1.0f);

            // ── Reactive mask ─────────────────────────────────────────────
            float reactive = reactiveBuffer ? reactiveBuffer[idx] : 0.0f;
            reactive = clamp(reactive, 0.0f, 1.0f);

            // Final: max of all three sources
            buf.preparedReactivity[idx] = std::max({maxTc, divergence, reactive});
        }
    }
}
