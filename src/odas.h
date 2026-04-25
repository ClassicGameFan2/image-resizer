// =============================================================================
// odas.h
// ODAS - Optimized Directional Anisotropic Sharpening
//
// Implementation of the algorithm described in:
// "An improved method for upscaling 2D images using adaptive edge sharpening
//  and an optimized directional anisotropic diffusion filter"
// Panda & Meher, Engineering Applications of AI, 136 (2024) 109045
//
// Pipeline:
//   LR Image
//     → Lanczos3 Upscale → F_hat
//     → Edge Detection (Canny) → F_E
//     → AES Filter (on F_E) → F_AES
//     → Smooth Region F_s = F_hat - F_E
//     → ODAD Filter (on F_s, λ from Cuckoo Search) → F_OTP
//     → IHR = F_AES + F_OTP
//     → BHR = Lanczos3_Downscale(Lanczos3_Upscale(IHR))
//     → F_RES = IHR - BHR
//     → F_RHR = IHR + F_RES  (final output)
// =============================================================================
#pragma once
#include <vector>
#include <cstdint>

// -----------------------------------------------------------------------------
// ODAS Parameters - all algorithm constants in one place
// -----------------------------------------------------------------------------
struct OdasParams {
    // Lanczos kernel
    int    lanczosQ        = 3;       // kernel size parameter (p=3 → 6x6 support)

    // Edge detection (Canny)
    float  cannyLowThresh  = 50.0f;   // Canny low threshold  (0-255 scale)
    float  cannyHighThresh = 150.0f;  // Canny high threshold (0-255 scale)

    // AES (Adaptive Edge Sharpening)
    // cmp levels are fixed by paper: {16, 24, 32, 40, 48}
    // Variance step S = (VR_max - VR_min) / 4  [computed per image]

    // ODAD filter
    float  K               = 0.20f;   // Edge-strength / diffusion threshold
    int    odadIterations  = 1;       // Fixed at 1 per paper
    // λ (lambda) is optimized by Cuckoo Search; range [0, π/4]

    // Cuckoo Search optimization
    int    csNests         = 25;      // Population size
    int    csMaxIter       = 100;     // Maximum iterations
    float  csPa            = 0.25f;   // Abandonment probability
    float  csLevyBeta      = 1.5f;    // Lévy flight exponent

    // Color handling
    // true  = process Y channel in YCbCr, bilinear upscale Cb/Cr
    // false = process each RGB channel independently
    bool   useYCbCr        = true;
};

// -----------------------------------------------------------------------------
// Main entry point
// upscales input (inW x inH, RGBA uint8) to output (outW x outH, RGBA uint8)
// -----------------------------------------------------------------------------
void scaleODAS(
    const unsigned char* input,  int inW,  int inH,
    unsigned char*       output, int outW, int outH,
    const OdasParams&    params = OdasParams{});
