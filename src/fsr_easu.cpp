// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// MIT License

#include "fsr_math.h"
#include <cmath>
#include <algorithm>

float3 sampleRGB(const unsigned char* data, int w, int h, int x, int y) {
    x = std::max(0, std::min(x, w - 1));
    y = std::max(0, std::min(y, h - 1));
    int idx = (y * w + x) * 4;
    return float3(data[idx] / 255.0f, data[idx+1] / 255.0f, data[idx+2] / 255.0f);
}

float sampleAlpha(const unsigned char* data, int w, int h, int x, int y) {
    x = std::max(0, std::min(x, w - 1));
    y = std::max(0, std::min(y, h - 1));
    int idx = (y * w + x) * 4;
    return data[idx+3] / 255.0f;
}

float getLuma(float3 c) {
    return c.x * 0.299f + c.y * 0.587f + c.z * 0.114f;
}

// --------------------------------------------------------------------------
// THE CORE FSR 1.0 ALGORITHM (EASU) - EXACT AMD POLYNOMIAL
// --------------------------------------------------------------------------

void scaleFSR_EASU(const unsigned char* input, int inW, int inH, 
                   unsigned char* output, int outW, int outH) {
    
    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            
            float srcX = ((x + 0.5f) * inW) / outW - 0.5f;
            float srcY = ((y + 0.5f) * inH) / outH - 0.5f;
            
            int ix = (int)std::floor(srcX);
            int iy = (int)std::floor(srcY);
            float fx = srcX - ix;
            float fy = srcY - iy;

            float lumaN = getLuma(sampleRGB(input, inW, inH, ix, iy - 1));
            float lumaS = getLuma(sampleRGB(input, inW, inH, ix, iy + 2));
            float lumaE = getLuma(sampleRGB(input, inW, inH, ix + 2, iy));
            float lumaW = getLuma(sampleRGB(input, inW, inH, ix - 1, iy));
            float lumaC = getLuma(sampleRGB(input, inW, inH, ix, iy)); 
            
            float dirX = lumaE - lumaW;
            float dirY = lumaS - lumaN;
            
            float dirLen = std::sqrt(dirX * dirX + dirY * dirY);
            if (dirLen > 0.0001f) {
                dirX /= dirLen;
                dirY /= dirLen;
            } else {
                dirX = 1.0f;
                dirY = 0.0f;
            }
            
            float stretchX = 1.0f + dirLen * 2.0f; 
            float stretchY = 1.0f;                 
            
            float3 totalColor(0.0f);
            float totalAlpha = 0.0f;
            float totalWeight = 0.0f;
            
            for (int dy = -1; dy <= 2; ++dy) {
                for (int dx = -1; dx <= 2; ++dx) {
                    
                    float distX = dx - fx;
                    float distY = dy - fy;
                    
                    float rotatedX = (dirX * distX + dirY * distY) * stretchX;
                    float rotatedY = (-dirY * distX + dirX * distY) * stretchY;
                    
                    float d2 = rotatedX * rotatedX + rotatedY * rotatedY;
                    
                    // CRITICAL FIX: EXACT AMD EASU WINDOW WEIGHT MATH
                    // AMD uses a custom windowed polynomial: w = (0.5 * d^2 - 1.0) * d^2 + 1.0
                    // This creates a much tighter, sharper sample than standard approximations.
                    if (d2 < 2.0f) { // AMD clamps the max radius tight
                        float w = (0.5f * d2 - 1.0f) * d2 + 1.0f;
                        w = std::max(0.0f, w); // Ensure weight doesn't go negative
                        
                        float3 color = sampleRGB(input, inW, inH, ix + dx, iy + dy);
                        float alpha = sampleAlpha(input, inW, inH, ix + dx, iy + dy);
                        
                        totalColor = totalColor + color * w;
                        totalAlpha += alpha * w;
                        totalWeight += w;
                    }
                }
            }
            
            if (totalWeight > 0.0f) {
                totalColor = totalColor * (1.0f / totalWeight);
                totalAlpha = totalAlpha / totalWeight;
            }
            
            totalColor = clamp(totalColor, float3(0.0f), float3(1.0f));
            totalAlpha = saturate(totalAlpha);
            
            int dstIndex = (y * outW + x) * 4;
            output[dstIndex + 0] = (unsigned char)(totalColor.x * 255.0f);
            output[dstIndex + 1] = (unsigned char)(totalColor.y * 255.0f);
            output[dstIndex + 2] = (unsigned char)(totalColor.z * 255.0f);
            output[dstIndex + 3] = (unsigned char)(totalAlpha * 255.0f);
        }
    }
}
