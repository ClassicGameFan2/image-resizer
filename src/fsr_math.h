#pragma once
#include <cmath>
#include <algorithm>

// --------------------------------------------------------
// CPU-GPU Math Bridge
// This file teaches standard C++ how to use HLSL Vector Math
// --------------------------------------------------------

struct float2 {
    float x, y;
    float2() : x(0), y(0) {}
    float2(float v) : x(v), y(v) {}
    float2(float x, float y) : x(x), y(y) {}
    
    float2 operator+(const float2& o) const { return float2(x + o.x, y + o.y); }
    float2 operator-(const float2& o) const { return float2(x - o.x, y - o.y); }
    float2 operator*(const float2& o) const { return float2(x * o.x, y * o.y); }
    float2 operator*(float s) const { return float2(x * s, y * s); }
    float2 operator-() const { return float2(-x, -y); }
};

struct float3 {
    float x, y, z;
    float3() : x(0), y(0), z(0) {}
    float3(float v) : x(v), y(v), z(v) {}
    float3(float x, float y, float z) : x(x), y(y), z(z) {}
    
    float3 operator+(const float3& o) const { return float3(x + o.x, y + o.y, z + o.z); }
    float3 operator-(const float3& o) const { return float3(x - o.x, y - o.y, z - o.z); }
    float3 operator*(const float3& o) const { return float3(x * o.x, y * o.y, z * o.z); }
    float3 operator*(float s) const { return float3(x * s, y * s, z * s); }
    float3 operator-() const { return float3(-x, -y, -z); }
};

struct float4 {
    float x, y, z, w;
    float4() : x(0), y(0), z(0), w(0) {}
    float4(float v) : x(v), y(v), z(v), w(v) {}
    float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    
    float4 operator+(const float4& o) const { return float4(x + o.x, y + o.y, z + o.z, w + o.w); }
    float4 operator-(const float4& o) const { return float4(x - o.x, y - o.y, z - o.z, w - o.w); }
    float4 operator*(const float4& o) const { return float4(x * o.x, y * o.y, z * o.z, w * o.w); }
    float4 operator*(float s) const { return float4(x * s, y * s, z * s, w * s); }
    float4 operator-() const { return float4(-x, -y, -z, -w); }
};

// --------------------------------------------------------
// HLSL GPU Math Functions
// --------------------------------------------------------

inline float min(float a, float b) { return std::min(a, b); }
inline float max(float a, float b) { return std::max(a, b); }

inline float clamp(float v, float minVal, float maxVal) { 
    return std::max(minVal, std::min(v, maxVal)); 
}

inline float saturate(float v) { 
    return clamp(v, 0.0f, 1.0f); 
}

inline float lerp(float a, float b, float t) { 
    return a + t * (b - a); 
}

inline float3 min(const float3& a, const float3& b) { 
    return float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); 
}
inline float3 max(const float3& a, const float3& b) { 
    return float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); 
}
inline float3 clamp(const float3& v, const float3& minVal, const float3& maxVal) {
    return float3(clamp(v.x, minVal.x, maxVal.x), clamp(v.y, minVal.y, maxVal.y), clamp(v.z, minVal.z, maxVal.z));
}
inline float3 saturate(const float3& v) {
    return float3(saturate(v.x), saturate(v.y), saturate(v.z));
}

inline float fract(float x) { 
    return x - std::floor(x); 
}

// ffxMax3 / ffxMin3 - used by RCAS
inline float ffxMax3(float a, float b, float c) { return max(max(a, b), c); }
inline float ffxMin3(float a, float b, float c) { return min(min(a, b), c); }

// ffxIsGreaterThanZero - returns 1.0 if x > 0, else 0.0 (matches AMD's AGtZeroF1)
inline float ffxIsGreaterThanZero(float x) { return x > 0.0f ? 1.0f : 0.0f; }
inline float3 ffxIsGreaterThanZero(const float3& v) {
    return float3(ffxIsGreaterThanZero(v.x), ffxIsGreaterThanZero(v.y), ffxIsGreaterThanZero(v.z));
}

// --------------------------------------------------------
// AMD POST-PROCESSING
// --------------------------------------------------------

// --------------------------------------------------------
// [LFGA] Linear Film Grain Applicator
// AMD FSR 1.2.2 - ffx_fsr1.h - FsrLfgaF
// Usage: pass color {0 to 1}, grain {-0.5 to 0.5}, amount {0 to 1}
// Grain is limited by distance to signal limits for temporal energy preservation.
// --------------------------------------------------------
inline float3 FsrLfgaF(float3 c, float3 t, float a) {
    // c += (t * a) * min(1.0 - c, c)
    float3 limit = float3(
        min(1.0f - c.x, c.x),
        min(1.0f - c.y, c.y),
        min(1.0f - c.z, c.z)
    );
    return float3(
        c.x + (t.x * a) * limit.x,
        c.y + (t.y * a) * limit.y,
        c.z + (t.z * a) * limit.z
    );
}

// --------------------------------------------------------
// [SRTM] Simple Reversible Tone-Mapper
// AMD FSR 1.2.2 - ffx_fsr1.h - FsrSrtmF / FsrSrtmInvF
// Converts linear HDR {0 to FP16_MAX} <-> {0 to 1}
// Preserves RGB ratio for HDR color bleed during filtering.
// NOTE: Not yet used in this app (no HDR pipeline), but ready for future use.
// --------------------------------------------------------
inline float3 FsrSrtmF(float3 c) {
    // c *= rcp(max3(c.r, c.g, c.b) + 1.0)
    float maxChannel = ffxMax3(c.x, c.y, c.z);
    float rcp = 1.0f / (maxChannel + 1.0f);
    return float3(c.x * rcp, c.y * rcp, c.z * rcp);
}

inline float3 FsrSrtmInvF(float3 c) {
    // c *= rcp(max(1/32768, 1.0 - max3(c.r, c.g, c.b)))
    // The extra max solves the c=1.0 case (which is a /0).
    float maxChannel = ffxMax3(c.x, c.y, c.z);
    float rcp = 1.0f / std::max(1.0f / 32768.0f, 1.0f - maxChannel);
    return float3(c.x * rcp, c.y * rcp, c.z * rcp);
}

// --------------------------------------------------------
// [TEPD] Temporal Energy Preserving Dither
// AMD FSR 1.2.2 - ffx_fsr1.h - FsrTepdDitF / FsrTepdC8F
// 8-bit gamma 2.0 dither. Chooses the linear nearest step
// point instead of perceptual nearest for temporal energy preservation.
// --------------------------------------------------------

// Dither value generator. Output is {0 to <1}.
// Uses the golden ratio for a good spatial distribution.
inline float FsrTepdDitF(int px, int py, int frame) {
    float x = (float)(px + frame);
    float y = (float)(py);
    // Golden ratio
    float a = (1.0f + std::sqrt(5.0f)) / 2.0f;
    // Pattern constant
    float b = 1.0f / 3.69f;
    x = x * a + (y * b);
    return fract(x);
}

// 8-bit gamma 2.0 TEPD conversion.
// Input 'c' is {0 to 1} linear. Output is {0 to 1} gamma-2.0 dithered.
inline float3 FsrTepdC8F(float3 c, float dit) {
    // n = floor(sqrt(c) * 255) / 255
    float3 n = float3(std::sqrt(c.x), std::sqrt(c.y), std::sqrt(c.z));
    n = float3(
        std::floor(n.x * 255.0f) * (1.0f / 255.0f),
        std::floor(n.y * 255.0f) * (1.0f / 255.0f),
        std::floor(n.z * 255.0f) * (1.0f / 255.0f)
    );
    // a = n^2 (lower gamma-2.0 boundary)
    float3 a = float3(n.x * n.x, n.y * n.y, n.z * n.z);
    // b = (n + 1/255)^2 (upper gamma-2.0 boundary)
    float3 b_ = float3(n.x + 1.0f / 255.0f, n.y + 1.0f / 255.0f, n.z + 1.0f / 255.0f);
    b_ = float3(b_.x * b_.x, b_.y * b_.y, b_.z * b_.z);
    // r = ratio of 'a' to 'b' required to produce 'c'
    // Uses safe reciprocal to avoid /0
    float3 r = float3(
        (c.x - b_.x) / std::max(a.x - b_.x, -1e-6f),
        (c.y - b_.y) / std::max(a.y - b_.y, -1e-6f),
        (c.z - b_.z) / std::max(a.z - b_.z, -1e-6f)
    );
    // Use ratio as cutoff: if dit > r, step up by 1/255
    float3 result = float3(
        n.x + ffxIsGreaterThanZero(dit - r.x) * (1.0f / 255.0f),
        n.y + ffxIsGreaterThanZero(dit - r.y) * (1.0f / 255.0f),
        n.z + ffxIsGreaterThanZero(dit - r.z) * (1.0f / 255.0f)
    );
    return saturate(result);
}

// --------------------------------------------------------
// applyPostProcess - applies LFGA and/or TEPD
// Called from EASU, RCAS, and standard scalers.
// --------------------------------------------------------
inline float3 applyPostProcess(float3 color, int x, int y, float lfga, bool tepd) {
    // 1. AMD Linear Film Grain Applicator (LFGA) - FSR 1.2.2
    if (lfga > 0.0f) {
        // Deterministic spatial noise in {-0.5 to 0.5} range
        float noise = fract(std::sin(x * 12.9898f + y * 78.233f) * 43758.5453f) - 0.5f;
        // Apply as monochrome grain: FsrLfgaF(color, float3(noise), lfga)
        color = FsrLfgaF(color, float3(noise, noise, noise), lfga);
    }
    
    // 2. AMD Temporal Energy Preserving Dither (TEPD) - FSR 1.2.2
    // Converts linear {0 to 1} to gamma-2.0 dithered {0 to 1}
    if (tepd) {
        float dit = FsrTepdDitF(x, y, 0);
        color = FsrTepdC8F(color, dit);
    }
    return color;
}
