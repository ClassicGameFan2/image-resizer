// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// MIT License

#include "fsr_math.h"
#include <cmath>
#include <algorithm>

extern float3 sampleRGB(const unsigned char* data, int w, int h, int x, int y);
extern float sampleAlpha(const unsigned char* data, int w, int h, int x, int y);

// --------------------------------------------------------------------------
// THE RCAS ALGORITHM (Robust Contrast Adaptive Sharpening) - FIXED
// --------------------------------------------------------------------------

void applyFSR_RCAS(const unsigned char* input, int w, int h, 
                   unsigned char* output, float sharpness) {
    
    // Convert sharpness to the math scaling limit.
    // The maximum safe weight for sharpening is strictly < 0.25 (to avoid dividing by zero).
    // AMD's formula allows sharpness from 0.0 (sharpest) to 2.0 (softest).
    float sharpnessLimit = 0.2f * std::exp2(-sharpness);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            float3 b = sampleRGB(input, w, h, x, y - 1); // North
            float3 d = sampleRGB(input, w, h, x - 1, y); // West
            float3 e = sampleRGB(input, w, h, x, y);     // Center
            float3 f = sampleRGB(input, w, h, x + 1, y); // East
            float3 h_ = sampleRGB(input, w, h, x, y + 1); // South

            float alpha = sampleAlpha(input, w, h, x, y);

            float3 minRGB = min(min(b, d), min(f, h_));
            minRGB = min(minRGB, e);
            
            float3 maxRGB = max(max(b, d), max(f, h_));
            maxRGB = max(maxRGB, e);

            float3 hitMin = minRGB;
            float3 hitMax = float3(1.0f) - maxRGB;
            float3 amz = min(hitMin, hitMax);
            
            float3 maxC = max(maxRGB, float3(0.00001f));
            float3 rz = float3(1.0f / maxC.x, 1.0f / maxC.y, 1.0f / maxC.z);
            
            // Calculate the adaptive lobe weight (0.0 to 1.0) based on contrast
            float3 w_adaptive = amz * rz;
            
            // Scale it by our sharpness limit
            float3 weight = w_adaptive * sharpnessLimit;
            
            // CRITICAL FIX: To SHARPEN, we must SUBTRACT the neighbors from the center!
            // We also subtract the weights from the denominator to keep the math balanced.
            float3 num = e - (b * weight + d * weight + f * weight + h_ * weight);
            float3 den = float3(1.0f) - float3(4.0f) * weight;
            float3 finalColor = num * float3(1.0f / den.x, 1.0f / den.y, 1.0f / den.z);

            finalColor = clamp(finalColor, float3(0.0f), float3(1.0f));

            int dstIndex = (y * w + x) * 4;
            output[dstIndex + 0] = (unsigned char)(finalColor.x * 255.0f);
            output[dstIndex + 1] = (unsigned char)(finalColor.y * 255.0f);
            output[dstIndex + 2] = (unsigned char)(finalColor.z * 255.0f);
            output[dstIndex + 3] = (unsigned char)(alpha * 255.0f);
        }
    }
}
