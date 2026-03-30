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
    float2 operator-() const { return float2(-x, -y); } // <-- FIX: Unary Minus
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
    float3 operator-() const { return float3(-x, -y, -z); } // <-- FIX: Unary Minus
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
    float4 operator-() const { return float4(-x, -y, -z, -w); } // <-- FIX: Unary Minus
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

inline float fract(float x) { 
    return x - std::floor(x); 
}

// --------------------------------------------------------
// AMD POST-PROCESSING (LFGA & TEPD)
// --------------------------------------------------------
inline float3 applyPostProcess(float3 color, int x, int y, float lfga, bool tepd) {
    // 1. AMD Linear Film Grain Applicator (LFGA)
    if (lfga > 0.0f) {
        // Fast, deterministic spatial noise generator
        float noise = fract(std::sin(x * 12.9898f + y * 78.233f) * 43758.5453f) - 0.5f;
        color.x += (noise * lfga) * std::min(1.0f - color.x, color.x);
        color.y += (noise * lfga) * std::min(1.0f - color.y, color.y);
        color.z += (noise * lfga) * std::min(1.0f - color.z, color.z);
    }
    
    // 2. AMD Temporal Energy Preserving Dither (TEPD)
    if (tepd) {
        float dit = fract(x * 1.61803398875f + y * 0.27100271f);
        auto applyDither = [&](float c) {
            float n = std::floor(c * 255.0f) * (1.0f / 255.0f);
            float r = (c - n) * 255.0f;
            return saturate(n + (dit < r ? (1.0f / 255.0f) : 0.0f));
        };
        color.x = applyDither(color.x);
        color.y = applyDither(color.y);
        color.z = applyDither(color.z);
    }
    return color;
}
