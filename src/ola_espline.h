// =============================================================================
// ola_espline.h
// OLA e-spline: Optimized Local Adaptive Edge-Preserving Spline Interpolation
//
// Based on: "An improved Image Interpolation technique using OLA e-spline"
//            by Jagyanseni Panda and Sukadev Meher
//
// Pipeline (Figure 2):
//   1. LA Gaussian filtering [Algorithm 1]  -> H_Ab (adaptive blur)
//   2. HPF extraction [Eq.4]               -> H(x,y) = G_LR - H_Ab
//   3. CS gain optimisation [Algorithm 2]  -> k
//   4. USM sharpening [Eq.1]              -> G_SLR = G_LR + k*H
//   5. Cubic B-spline upscale [Eq.6,7]    -> G_HR
//   6. Canny edge detection               -> edge map
//   7. e-spline edge expansion [Eq.9-14]  -> G_e (delta)
//   8. Final combination [Eq.15]          -> G_RHR = G_HR + G_e
// =============================================================================

#pragma once
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
struct OLAESplineParams
{
    // Upscale
    int   scaleFactor = 2;      // integer scale factor (2 or 4)

    // CS optimisation (Algorithm 2)
    int    csNests    = 25;     // number of host nests
    int    csMaxGen   = 50;     // maximum generations
    double csPa       = 0.25;   // alien-egg discovery probability
    double csBeta     = 1.5;    // Levy exponent, 1 < beta < 3
    double csKMin     = 0.0;    // gain k search lower bound
    double csKMax     = 3.0;    // gain k search upper bound
    unsigned csSeed   = 42;     // RNG seed

    // Fitness subsampling: evaluate every N-th pixel to speed up CS.
    // 1 = exact (slow on large images), 4 = fast, 8 = very fast.
    int    csFitnessStep = 4;

    // Canny edge detection
    float  cannyLow   = 20.f;
    float  cannyHigh  = 50.f;
};

// ---------------------------------------------------------------------------
// Main entry point — operates on RGBA uint8 data (channels processed
// independently as the paper evaluates on single-channel metrics).
// Output buffer must be pre-allocated: outW * outH * 4 bytes.
// outW = inW * scaleFactor, outH = inH * scaleFactor.
// ---------------------------------------------------------------------------
void scaleOLAESpline(const unsigned char* input,  int inW,  int inH,
                           unsigned char* output, int outW, int outH,
                     const OLAESplineParams& params = OLAESplineParams{});
