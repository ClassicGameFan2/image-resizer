// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

#include "fsr_math.h"
#include <cmath>
#include <algorithm>

// --------------------------------------------------------------------------
// CPU TEXTURE SAMPLERS
// These mimic the GPU hardware, safely reading pixels and converting them 
// from 8-bit whole numbers (0-255) into GPU floats (0.0 to 1.0)
// --------------------------------------------------------------------------

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

// Calculate image brightness (Luma). FSR uses this to detect edges.
float getLuma(float3 c) {
    return c.x * 0.299f + c.y * 0.587f + c.z * 0.114f;
}

// --------------------------------------------------------------------------
// THE CORE FSR 1.0 ALGORITHM (EASU)
// --------------------------------------------------------------------------

void scaleFSR_EASU(const unsigned char* input, int inW, int inH, 
                   unsigned char* output, int outW, int outH) {
    
    // Process every single pixel of the NEW, larger image
    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            
            // 1. Map the large image pixel back to the original small image.
            // We add 0.5 to target the exact "center" of the pixel.
            float srcX = ((x + 0.5f) * inW) / outW - 0.5f;
            float srcY = ((y + 0.5f) * inH) / outH - 0.5f;
            
            // Get the integer top-left coordinate and the fractional offset
            int ix = (int)std::floor(srcX);
            int iy = (int)std::floor(srcY);
            float fx = srcX - ix;
            float fy = srcY - iy;

            // 2. Sample a cross-pattern to detect contrast and edges
            float lumaN = getLuma(sampleRGB(input, inW, inH, ix, iy - 1));
            float lumaS = getLuma(sampleRGB(input, inW, inH, ix, iy + 2));
            float lumaE = getLuma(sampleRGB(input, inW, inH, ix + 2, iy));
            float lumaW = getLuma(sampleRGB(input, inW, inH, ix - 1, iy));
            float lumaC = getLuma(sampleRGB(input, inW, inH, ix, iy)); // Center
            
            // 3. Calculate the direction of the edge (Gradient)
            float dirX = lumaE - lumaW;
            float dirY = lumaS - lumaN;
            
            // Normalize direction and measure how strong the edge is
            float dirLen = std::sqrt(dirX * dirX + dirY * dirY);
            if (dirLen > 0.0001f) {
                dirX /= dirLen;
                dirY /= dirLen;
            } else {
                dirX = 1.0f;
                dirY = 0.0f;
            }
            
            // FSR stretches the sampling window along the edge to keep it sharp,
            // while slightly blurring parallel to the edge to remove jaggies.
            float stretchX = 1.0f + dirLen * 2.0f; 
            float stretchY = 1.0f;                 
            
            // 4. Gather the 4x4 Grid of pixels and apply the FSR Weights
            float3 totalColor(0.0f);
            float totalAlpha = 0.0f;
            float totalWeight = 0.0f;
            
            // Loop through a 4x4 box around our target point
            for (int dy = -1; dy <= 2; ++dy) {
                for (int dx = -1; dx <= 2; ++dx) {
                    
                    float distX = dx - fx;
                    float distY = dy - fy;
                    
                    // Rotate and stretch the distance based on the edge direction
                    float rotatedX = (dirX * distX + dirY * distY) * stretchX;
                    float rotatedY = (-dirY * distX + dirX * distY) * stretchY;
                    
                    // FSR's Custom Window Math (Similar to Lanczos)
                    float d2 = rotatedX * rotatedX + rotatedY * rotatedY;
                    
                    // If the pixel is inside the sampling radius, calculate its weight
                    if (d2 < 4.0f) {
                        float w = std::exp2(-d2 * 2.0f); // Fast exponential curve
                        
                        float3 color = sampleRGB(input, inW, inH, ix + dx, iy + dy);
                        float alpha = sampleAlpha(input, inW, inH, ix + dx, iy + dy);
                        
                        totalColor = totalColor + color * w;
                        totalAlpha += alpha * w;
                        totalWeight += w;
                    }
                }
            }
            
            // 5. Normalize the final gathered colors
            if (totalWeight > 0.0f) {
                totalColor = totalColor * (1.0f / totalWeight);
                totalAlpha = totalAlpha / totalWeight;
            }
            
            // Ensure the colors don't accidentally exceed the 0.0 - 1.0 limits
            totalColor = clamp(totalColor, float3(0.0f), float3(1.0f));
            totalAlpha = saturate(totalAlpha);
            
            // 6. Convert back to standard 8-bit bytes and save to memory
            int dstIndex = (y * outW + x) * 4;
            output[dstIndex + 0] = (unsigned char)(totalColor.x * 255.0f);
            output[dstIndex + 1] = (unsigned char)(totalColor.y * 255.0f);
            output[dstIndex + 2] = (unsigned char)(totalColor.z * 255.0f);
            output[dstIndex + 3] = (unsigned char)(totalAlpha * 255.0f);
        }
    }
}
