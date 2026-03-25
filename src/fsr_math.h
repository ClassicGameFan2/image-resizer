#pragma once
#include <cmath>
#include <algorithm>

// --------------------------------------------------------
// CPU-GPU Math Bridge
// This file teaches standard C++ how to use HLSL Vector Math
// --------------------------------------------------------

// A 2-part vector (usually used for X, Y coordinates)
struct float2 {
    float x, y;
    float2() : x(0), y(0) {}
    float2(float v) : x(v), y(v) {}
    float2(float x, float y) : x(x), y(y) {}
    
    float2 operator+(const float2& o) const { return float2(x + o.x, y + o.y); }
    float2 operator-(const float2& o) const { return float2(x - o.x, y - o.y); }
    float2 operator*(const float2& o) const { return float2(x * o.x, y * o.y); }
    float2 operator*(float s) const { return float2(x * s, y * s); }
};

// A 3-part vector (usually used for Red, Green, Blue colors)
struct float3 {
    float x, y, z;
    float3() : x(0), y(0), z(0) {}
    float3(float v) : x(v), y(v), z(v) {}
    float3(float x, float y, float z) : x(x), y(y), z(z) {}
    
    float3 operator+(const float3& o) const { return float3(x + o.x, y + o.y, z + o.z); }
    float3 operator-(const float3& o) const { return float3(x - o.x, y - o.y, z - o.z); }
    float3 operator*(const float3& o) const { return float3(x * o.x, y * o.y, z * o.z); }
    float3 operator*(float s) const { return float3(x * s, y * s, z * s); }
};

// A 4-part vector (usually used for Red, Green, Blue, Alpha)
struct float4 {
    float x, y, z, w;
    float4() : x(0), y(0), z(0), w(0) {}
    float4(float v) : x(v), y(v), z(v), w(v) {}
    float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    
    float4 operator+(const float4& o) const { return float4(x + o.x, y + o.y, z + o.z, w + o.w); }
    float4 operator-(const float4& o) const { return float4(x - o.x, y - o.y, z - o.z, w - o.w); }
    float4 operator*(const float4& o) const { return float4(x * o.x, y * o.y, z * o.z, w * o.w); }
    float4 operator*(float s) const { return float4(x * s, y * s, z * s, w * s); }
};

// --------------------------------------------------------
// HLSL GPU Math Functions
// --------------------------------------------------------

inline float min(float a, float b) { return std::min(a, b); }
inline float max(float a, float b) { return std::max(a, b); }

// Keeps a number strictly between a minimum and maximum value
inline float clamp(float v, float minVal, float maxVal) { 
    return std::max(minVal, std::min(v, maxVal)); 
}

// "Saturate" is a GPU term. It means clamp between 0.0 (Black) and 1.0 (White).
inline float saturate(float v) { 
    return clamp(v, 0.0f, 1.0f); 
}

// "Lerp" means Linear Interpolation. It blends two values together smoothly.
inline float lerp(float a, float b, float t) { 
    return a + t * (b - a); 
}

// We need vector versions of min and max so the FSR math can process 3 colors at once!
inline float3 min(const float3& a, const float3& b) { 
    return float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); 
}
inline float3 max(const float3& a, const float3& b) { 
    return float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); 
}
inline float3 clamp(const float3& v, const float3& minVal, const float3& maxVal) {
    return float3(clamp(v.x, minVal.x, maxVal.x), clamp(v.y, minVal.y, maxVal.y), clamp(v.z, minVal.z, maxVal.z));
}
