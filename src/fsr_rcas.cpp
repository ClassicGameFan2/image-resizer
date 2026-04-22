// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// MIT License
// Updated to FSR 1.2.2 (FidelityFX SDK v2.2.0)
// Key fix: "Fix for possible negative RCAS output" via lowerLimiterMultiplier.

#include "fsr_math.h"
#include <cmath>
#include <algorithm>

extern float3 sampleRGB(const unsigned char* data, int w, int h, int x, int y);
extern float sampleAlpha(const unsigned char* data, int w, int h, int x, int y);
extern float getLuma(float3 c);

#define FSR_RCAS_LIMIT (0.25f - (1.0f / 16.0f))

// --------------------------------------------------------------------------
// AMD RCAS ALGORITHM - FSR 1.2.2
// Key change from 1.0.2: Added lowerLimiterMultiplier to fix possible
// negative output. hitMin now uses mn4 directly (not min(mn4, e)).
// --------------------------------------------------------------------------
void applyFSR_RCAS(const unsigned char* input, int w, int h, unsigned char* output, 
                   float sharpness, bool useDenoise, float lfga, bool useTepd) {

    // Convert sharpness from stops to linear value.
    float sharpConfig = std::exp2(-sharpness);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            float3 b  = sampleRGB(input, w, h, x,     y - 1);
            float3 d  = sampleRGB(input, w, h, x - 1, y    );
            float3 e  = sampleRGB(input, w, h, x,     y    );
            float3 f  = sampleRGB(input, w, h, x + 1, y    );
            float3 h_ = sampleRGB(input, w, h, x,     y + 1);

            // Luma times 2 (matches AMD's luma formula exactly)
            float bL = b.z * 0.5f + (b.x * 0.5f + b.y);
            float dL = d.z * 0.5f + (d.x * 0.5f + d.y);
            float eL = e.z * 0.5f + (e.x * 0.5f + e.y);
            float fL = f.z * 0.5f + (f.x * 0.5f + f.y);
            float hL = h_.z * 0.5f + (h_.x * 0.5f + h_.y);

            // Noise detection (unchanged from 1.0.2)
            float nz = 0.25f * bL + 0.25f * dL + 0.25f * fL + 0.25f * hL - eL;
            float maxL = ffxMax3(ffxMax3(bL, dL, eL), fL, hL);
            float minL = ffxMin3(ffxMin3(bL, dL, eL), fL, hL);
            nz = saturate(std::abs(nz) / std::max(maxL - minL, 1e-6f));
            nz = -0.5f * nz + 1.0f;

            // Min/max of ring (unchanged)
            float3 mn4 = min(min(min(b, d), f), h_);
            float3 mx4 = max(max(max(b, d), f), h_);

            // FSR 1.2.2 FIX: lowerLimiterMultiplier
            // Prevents possible negative output by scaling down the hitMin
            // contribution when eL is much smaller than the ring minimum luma.
            // saturate(eL / min(min3(bL, dL, fL), hL))
            float ringMinL = std::min(ffxMin3(bL, dL, fL), hL);
            float lowerLimiterMultiplier = saturate(eL / std::max(ringMinL, 1e-6f));

            // Immediate constants for peak range
            // peakC = float2(1.0, -4.0)

            // FSR 1.2.2 Limiters:
            // hitMin = mn4 * rcp(4.0 * mx4) * lowerLimiterMultiplier  [mn4 only, no 'e' term]
            // hitMax = (1.0 - mx4) * rcp(4.0 * mn4 - 4.0)
            float3 hitMin = float3(
                mn4.x / std::max(4.0f * mx4.x, 1e-6f) * lowerLimiterMultiplier,
                mn4.y / std::max(4.0f * mx4.y, 1e-6f) * lowerLimiterMultiplier,
                mn4.z / std::max(4.0f * mx4.z, 1e-6f) * lowerLimiterMultiplier
            );
            float3 hitMax = float3(
                (1.0f - mx4.x) / std::min(4.0f * mn4.x - 4.0f, -1e-6f),
                (1.0f - mx4.y) / std::min(4.0f * mn4.y - 4.0f, -1e-6f),
                (1.0f - mx4.z) / std::min(4.0f * mn4.z - 4.0f, -1e-6f)
            );

            float3 lobeRGB = max(-hitMin, hitMax);
            float lobe = std::max(
                -FSR_RCAS_LIMIT,
                std::min(ffxMax3(lobeRGB.x, lobeRGB.y, lobeRGB.z), 0.0f)
            ) * sharpConfig;

            // Optional RCAS denoise
            if (useDenoise) lobe *= nz;

            // Resolve
            float rcpL = 1.0f / (4.0f * lobe + 1.0f);
            float3 finalColor = (b * lobe + d * lobe + h_ * lobe + f * lobe + e) * rcpL;
            finalColor = clamp(finalColor, float3(0.0f), float3(1.0f));
            
            // Apply post-processing
            finalColor = applyPostProcess(finalColor, x, y, lfga, useTepd);
            
            float alpha = sampleAlpha(input, w, h, x, y);

            int dstIndex = (y * w + x) * 4;
            output[dstIndex + 0] = (unsigned char)(finalColor.x * 255.0f + 0.5f);
            output[dstIndex + 1] = (unsigned char)(finalColor.y * 255.0f + 0.5f);
            output[dstIndex + 2] = (unsigned char)(finalColor.z * 255.0f + 0.5f);
            output[dstIndex + 3] = (unsigned char)(alpha * 255.0f + 0.5f);
        }
    }
}
