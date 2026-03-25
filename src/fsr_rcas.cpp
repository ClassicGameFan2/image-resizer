// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// MIT License

#include "fsr_math.h"
#include <cmath>
#include <algorithm>

// Re-use our CPU Samplers
extern float3 sampleRGB(const unsigned char* data, int w, int h, int x, int y);
extern float sampleAlpha(const unsigned char* data, int w, int h, int x, int y);

// --------------------------------------------------------------------------
// THE RCAS ALGORITHM (Robust Contrast Adaptive Sharpening)
// --------------------------------------------------------------------------

void applyFSR_RCAS(const unsigned char* input, int w, int h, 
                   unsigned char* output, float sharpness) {
    
    // Convert the user's sharpness (0.0 to 2.0) into the math variable AMD uses.
    // 0.0 is maximum sharpness. 2.0 is minimum sharpness.
    float sharpnessConfig = std::exp2(-sharpness);

    // Process every single pixel of the ALREADY SCALED image
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            // 1. Sample the current pixel, and its 4 immediate neighbors (Cross pattern)
            float3 b = sampleRGB(input, w, h, x, y - 1); // North
            float3 d = sampleRGB(input, w, h, x - 1, y); // West
            float3 e = sampleRGB(input, w, h, x, y);     // Center
            float3 f = sampleRGB(input, w, h, x + 1, y); // East
            float3 h_ = sampleRGB(input, w, h, x, y + 1); // South

            // Pass the alpha straight through (we don't sharpen transparency)
            float alpha = sampleAlpha(input, w, h, x, y);

            // 2. Find the darkest and brightest colors in this cross pattern
            float3 minRGB = min(min(b, d), min(f, h_));
            minRGB = min(minRGB, e);
            
            float3 maxRGB = max(max(b, d), max(f, h_));
            maxRGB = max(maxRGB, e);

            // 3. RCAS Core Math: Calculate how much we are allowed to sharpen
            // This prevents "ringing" (ugly white halos around dark edges)
            float3 hitMin = minRGB;
            float3 hitMax = float3(1.0f) - maxRGB;
            float3 amz = min(hitMin, hitMax);
            
            // Avoid division by zero
            float3 maxC = max(maxRGB, float3(0.00001f));
            float3 rz = float3(1.0f / maxC.x, 1.0f / maxC.y, 1.0f / maxC.z);
            
            // Calculate the exact weight to apply to the neighbors
            float3 amzRz = amz * rz;
            float3 weight = amzRz * sharpnessConfig;
            
            // 4. Blend the sharpened color together
            float3 num = b * weight + d * weight + f * weight + h_ * weight + e;
            float3 den = float3(4.0f) * weight + float3(1.0f);
            float3 finalColor = num * float3(1.0f / den.x, 1.0f / den.y, 1.0f / den.z);

            // Keep within valid 0.0 - 1.0 limits
            finalColor = clamp(finalColor, float3(0.0f), float3(1.0f));

            // 5. Convert back to 8-bit bytes and save
            int dstIndex = (y * w + x) * 4;
            output[dstIndex + 0] = (unsigned char)(finalColor.x * 255.0f);
            output[dstIndex + 1] = (unsigned char)(finalColor.y * 255.0f);
            output[dstIndex + 2] = (unsigned char)(finalColor.z * 255.0f);
            output[dstIndex + 3] = (unsigned char)(alpha * 255.0f);
        }
    }
}
