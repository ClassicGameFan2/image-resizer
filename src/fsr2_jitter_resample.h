// =============================================================================
// fsr2_jitter_resample.h
// FSR 2.3.4 CPU Port — Sub-pixel jitter frame generator
// Adapted from the 3D testbed project's jitter_resample code.
// Creates synthetic jittered low-res frames from a static image.
// =============================================================================
#pragma once
#include <vector>

enum class Fsr2JitterMode { BILINEAR, BICUBIC, LANCZOS3 };

// Resample `src` (srcW x srcH, float RGBA) applying sub-pixel shift (shiftX, shiftY)
// into `dst` (same dimensions, float RGBA).
// shiftX/Y are in pixel units. Positive X shifts content right.
void fsr2ResampleWithShift(
    const float* src, int srcW, int srcH,
    float shiftX, float shiftY,
    float* dst,
    Fsr2JitterMode mode = Fsr2JitterMode::BICUBIC);
