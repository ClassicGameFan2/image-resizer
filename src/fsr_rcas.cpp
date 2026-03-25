// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// MIT License

#include "fsr_math.h"
#include <cmath>
#include <algorithm>

extern float3 sampleRGB(const unsigned char* data, int w, int h, int x, int y);
extern float sampleAlpha(const unsigned char* data, int w, int h, int x, int y);
extern float getLuma(float3 c);

#define FSR_RCAS_LIMIT (0.25f - (1.0f / 16.0f))

// --------------------------------------------------------------------------
// EXACT AMD RCAS ALGORITHM (FsrRcasF)
// --------------------------------------------------------------------------
void applyFSR_RCAS(const unsigned char* input, int w, int h, 
                   unsigned char* output, float sharpness) {
    
    // Convert sharpness from stops to linear value.
    float sharpConfig = std::exp2(-sharpness);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            float3 b = sampleRGB(input, w, h, x, y - 1);
            float3 d = sampleRGB(input, w, h, x - 1, y);
            float3 e = sampleRGB(input, w, h, x, y);
            float3 f = sampleRGB(input, w, h, x + 1, y);
            float3 h_ = sampleRGB(input, w, h, x, y + 1);

            float bL = getLuma(b); float dL = getLuma(d); float eL = getLuma(e); 
            float fL = getLuma(f); float hL = getLuma(h_);

            // EXACT AMD NOISE DETECTION
            float nz = 0.25f * bL + 0.25f * dL + 0.25f * fL + 0.25f * hL - eL;
            float maxL = std::max(std::max(std::max(bL, dL), eL), std::max(fL, hL));
            float minL = std::min(std::min(std::min(bL, dL), eL), std::min(fL, hL));
            nz = saturate(std::abs(nz) / std::max(maxL - minL, 1e-6f));
            nz = -0.5f * nz + 1.0f;

            // MIN/MAX RING
            float3 mn4 = min(min(b, d), min(f, h_));
            float3 mx4 = max(max(b, d), max(f, h_));

            // EXACT AMD LIMITERS
            float3 hitMin = min(mn4, e) * float3(1.0f / std::max(4.0f * mx4.x, 1e-6f),
                                                 1.0f / std::max(4.0f * mx4.y, 1e-6f),
                                                 1.0f / std::max(4.0f * mx4.z, 1e-6f));
                                                 
            float3 hitMax = (float3(1.0f) - max(mx4, e)) * float3(1.0f / std::min(4.0f * mn4.x - 4.0f, -1e-6f),
                                                                  1.0f / std::min(4.0f * mn4.y - 4.0f, -1e-6f),
                                                                  1.0f / std::min(4.0f * mn4.z - 4.0f, -1e-6f));
            
            float3 lobeRGB = max(-hitMin, hitMax);
            float lobe = std::max(-FSR_RCAS_LIMIT, std::min(std::max(lobeRGB.x, std::max(lobeRGB.y, lobeRGB.z)), 0.0f)) * sharpConfig;

            // Apply noise removal
            lobe *= nz;

            // Resolve
            float rcpL = 1.0f / (4.0f * lobe + 1.0f);
            float3 finalColor = (b * lobe + d * lobe + h_ * lobe + f * lobe + e) * rcpL;
            
            finalColor = clamp(finalColor, float3(0.0f), float3(1.0f));
            float alpha = sampleAlpha(input, w, h, x, y);

            int dstIndex = (y * w + x) * 4;
            output[dstIndex + 0] = (unsigned char)(finalColor.x * 255.0f);
            output[dstIndex + 1] = (unsigned char)(finalColor.y * 255.0f);
            output[dstIndex + 2] = (unsigned char)(finalColor.z * 255.0f);
            output[dstIndex + 3] = (unsigned char)(alpha * 255.0f);
        }
    }
}
