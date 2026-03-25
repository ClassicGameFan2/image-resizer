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

// EXACT AMD FSR LUMA FORMULA
float getLuma(float3 c) {
    return c.x * 0.5f + c.y * 1.0f + c.z * 0.5f;
}

// --------------------------------------------------------------------------
// EXACT AMD EASU EDGE ACCUMULATION (FsrEasuSetF)
// --------------------------------------------------------------------------
void FsrEasuSetF(float2& dir, float& len, float2 pp, bool biS, bool biT, bool biU, bool biV, 
                 float lA, float lB, float lC, float lD, float lE) {
    float w = 0.0f;
    if (biS) w = (1.0f - pp.x) * (1.0f - pp.y);
    if (biT) w = pp.x * (1.0f - pp.y);
    if (biU) w = (1.0f - pp.x) * pp.y;
    if (biV) w = pp.x * pp.y;

    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = std::max(std::abs(dc), std::abs(cb));
    lenX = 1.0f / std::max(lenX, 1e-6f); // Protect against /0
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = clamp(std::abs(dirX) * lenX, 0.0f, 1.0f);
    lenX *= lenX;
    len += lenX * w;

    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = std::max(std::abs(ec), std::abs(ca));
    lenY = 1.0f / std::max(lenY, 1e-6f);
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = clamp(std::abs(dirY) * lenY, 0.0f, 1.0f);
    lenY *= lenY;
    len += lenY * w;
}

// --------------------------------------------------------------------------
// EXACT AMD EASU TAP FILTER (FsrEasuTapF)
// --------------------------------------------------------------------------
void FsrEasuTapF(float3& aC, float& aW, float2 off, float2 dir, float2 len2, float lob, float clp, float3 c) {
    float2 v;
    v.x = (off.x * dir.x) + (off.y * dir.y);
    v.y = (off.x * -dir.y) + (off.y * dir.x);
    v.x *= len2.x;
    v.y *= len2.y;
    
    float d2 = v.x * v.x + v.y * v.y;
    d2 = std::min(d2, clp);
    
    float wB = (2.0f / 5.0f) * d2 - 1.0f;
    float wA = lob * d2 - 1.0f;
    wB *= wB;
    wA *= wA;
    wB = (25.0f / 16.0f) * wB - (25.0f / 16.0f - 1.0f);
    
    float w = wB * wA;
    aC = aC + c * w;
    aW += w;
}

// --------------------------------------------------------------------------
// THE CORE FSR 1.0 ALGORITHM (FsrEasuF)
// --------------------------------------------------------------------------
void scaleFSR_EASU(const unsigned char* input, int inW, int inH, 
                   unsigned char* output, int outW, int outH) {
    
    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            
            float srcX = ((x + 0.5f) * inW) / outW - 0.5f;
            float srcY = ((y + 0.5f) * inH) / outH - 0.5f;
            int ix = (int)std::floor(srcX);
            int iy = (int)std::floor(srcY);
            float2 pp = float2(srcX - ix, srcY - iy);

            // Fetch the 12-tap grid
            float3 b = sampleRGB(input, inW, inH, ix,   iy-1);
            float3 c = sampleRGB(input, inW, inH, ix+1, iy-1);
            float3 e = sampleRGB(input, inW, inH, ix-1, iy);
            float3 f = sampleRGB(input, inW, inH, ix,   iy);
            float3 g = sampleRGB(input, inW, inH, ix+1, iy);
            float3 h = sampleRGB(input, inW, inH, ix+2, iy);
            float3 i = sampleRGB(input, inW, inH, ix-1, iy+1);
            float3 j = sampleRGB(input, inW, inH, ix,   iy+1);
            float3 k = sampleRGB(input, inW, inH, ix+1, iy+1);
            float3 l = sampleRGB(input, inW, inH, ix+2, iy+1);
            float3 n = sampleRGB(input, inW, inH, ix,   iy+2);
            float3 o = sampleRGB(input, inW, inH, ix+1, iy+2);

            float bL = getLuma(b); float cL = getLuma(c);
            float eL = getLuma(e); float fL = getLuma(f); float gL = getLuma(g); float hL = getLuma(h);
            float iL = getLuma(i); float jL = getLuma(j); float kL = getLuma(k); float lL = getLuma(l);
            float nL = getLuma(n); float oL = getLuma(o);

            float2 dir(0.0f);
            float len = 0.0f;
            FsrEasuSetF(dir, len, pp, true, false, false, false, bL, eL, fL, gL, jL);
            FsrEasuSetF(dir, len, pp, false, true, false, false, cL, fL, gL, hL, kL);
            FsrEasuSetF(dir, len, pp, false, false, true, false, fL, iL, jL, kL, nL);
            FsrEasuSetF(dir, len, pp, false, false, false, true, gL, jL, kL, lL, oL);

            float dirR = dir.x * dir.x + dir.y * dir.y;
            bool zro = dirR < (1.0f / 32768.0f);
            dirR = 1.0f / std::sqrt(std::max(dirR, 1e-6f));
            dirR = zro ? 1.0f : dirR;
            dir.x = zro ? 1.0f : dir.x;
            dir.x *= dirR;
            dir.y *= dirR;

            len = len * 0.5f;
            len *= len;
            float stretch = (dir.x * dir.x + dir.y * dir.y) / std::max(std::max(std::abs(dir.x), std::abs(dir.y)), 1e-6f);
            float2 len2 = float2(1.0f + (stretch - 1.0f) * len, 1.0f - 0.5f * len);
            float lob = 0.5f + ((1.0f / 4.0f - 0.04f) - 0.5f) * len;
            float clp = 1.0f / lob;

            float3 min4 = min(min(f, g), min(j, k));
            float3 max4 = max(max(f, g), max(j, k));

            float3 aC(0.0f);
            float aW = 0.0f;
            FsrEasuTapF(aC, aW, float2( 0.0f, -1.0f) - pp, dir, len2, lob, clp, b);
            FsrEasuTapF(aC, aW, float2( 1.0f, -1.0f) - pp, dir, len2, lob, clp, c);
            FsrEasuTapF(aC, aW, float2(-1.0f,  0.0f) - pp, dir, len2, lob, clp, e);
            FsrEasuTapF(aC, aW, float2( 0.0f,  0.0f) - pp, dir, len2, lob, clp, f);
            FsrEasuTapF(aC, aW, float2( 1.0f,  0.0f) - pp, dir, len2, lob, clp, g);
            FsrEasuTapF(aC, aW, float2( 2.0f,  0.0f) - pp, dir, len2, lob, clp, h);
            FsrEasuTapF(aC, aW, float2(-1.0f,  1.0f) - pp, dir, len2, lob, clp, i);
            FsrEasuTapF(aC, aW, float2( 0.0f,  1.0f) - pp, dir, len2, lob, clp, j);
            FsrEasuTapF(aC, aW, float2( 1.0f,  1.0f) - pp, dir, len2, lob, clp, k);
            FsrEasuTapF(aC, aW, float2( 2.0f,  1.0f) - pp, dir, len2, lob, clp, l);
            FsrEasuTapF(aC, aW, float2( 0.0f,  2.0f) - pp, dir, len2, lob, clp, n);
            FsrEasuTapF(aC, aW, float2( 1.0f,  2.0f) - pp, dir, len2, lob, clp, o);

            float3 finalColor = aC * (1.0f / std::max(aW, 1e-6f));
            finalColor = clamp(finalColor, min4, max4);
            float alpha = sampleAlpha(input, inW, inH, ix, iy); // Center Alpha

            int dstIndex = (y * outW + x) * 4;
            output[dstIndex + 0] = (unsigned char)(finalColor.x * 255.0f);
            output[dstIndex + 1] = (unsigned char)(finalColor.y * 255.0f);
            output[dstIndex + 2] = (unsigned char)(finalColor.z * 255.0f);
            output[dstIndex + 3] = (unsigned char)(alpha * 255.0f);
        }
    }
}
