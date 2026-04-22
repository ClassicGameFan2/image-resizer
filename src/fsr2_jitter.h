// =============================================================================
// fsr2_jitter.h
// FSR 2.3.4 CPU Port — Halton(2,3) jitter sequence generator
// Ported from: ffx_fsr2.cpp (ffxFsr2GetJitterOffset, ffxFsr2GetJitterPhaseCount)
// AMD MIT License
// =============================================================================
#pragma once
#include <cstdint>
#include <cmath>

// ----------------------------------------------------------------
// Halton low-discrepancy sequence
// base=2 for X, base=3 for Y (matches AMD SDK exactly)
// ----------------------------------------------------------------
inline float halton(int index, int base) {
    float result = 0.0f;
    float f = 1.0f;
    int i = index;
    while (i > 0) {
        f /= (float)base;
        result += f * (float)(i % base);
        i = i / base;
    }
    return result;
}

// ----------------------------------------------------------------
// ffxFsr2GetJitterPhaseCount
// Returns the recommended jitter sequence length for a given
// render->display scale ratio.
// Formula: ceil(8.0 * (displayWidth / renderWidth)^2)
// ----------------------------------------------------------------
inline int fsr2GetJitterPhaseCount(int renderWidth, int displayWidth) {
    float ratio = (float)displayWidth / (float)renderWidth;
    return (int)std::ceil(8.0f * ratio * ratio);
}

// ----------------------------------------------------------------
// ffxFsr2GetJitterOffset
// Returns jitter offset in unit pixel space [-0.5 .. +0.5].
// index   = frameIndex % phaseCount
// phaseCount = fsr2GetJitterPhaseCount(...)
//
// The Halton sequence starts at index=1 (not 0!) so we add 1.
// This matches AMD's implementation exactly and avoids null vector.
// ----------------------------------------------------------------
inline void fsr2GetJitterOffset(float* outX, float* outY, int index, int phaseCount) {
    // AMD uses 1-based Halton: index starts at 1
    int haltonIndex = (index % phaseCount) + 1;
    // Output is in [-0.5 .. +0.5] pixel space
    *outX = halton(haltonIndex, 2) - 0.5f;
    *outY = halton(haltonIndex, 3) - 0.5f;
}
