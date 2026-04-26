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
    int   lanczosQ        = 3;
    float cannyLowThresh  = 50.0f;
    float cannyHighThresh = 150.0f;

    // K in [0,255] scale. Paper uses [0,1] so K_paper=0.2 → K_ours=51.0
    float K               = 51.0f;

    int   odadIterations  = 1;

    // CS parameters — more nests/iterations = better λ search
    int   csNests         = 25;
    int   csMaxIter       = 100;
    float csPa            = 0.25f;
    float csLevyBeta      = 1.5f;

    bool  useYCbCr        = true;
};

// -----------------------------------------------------------------------------
// Main entry point
// upscales input (inW x inH, RGBA uint8) to output (outW x outH, RGBA uint8)
// -----------------------------------------------------------------------------
void scaleODAS(
    const unsigned char* input,  int inW,  int inH,
    unsigned char*       output, int outW, int outH,
    const OdasParams&    params = OdasParams{});
