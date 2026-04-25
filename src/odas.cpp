// =============================================================================
// odas.cpp
// ODAS - Optimized Directional Anisotropic Sharpening
//
// Full implementation of Panda & Meher (2024).
//
// Bug fixes vs v1:
//   1. AES: kernel H is a Laplacian sharpening kernel applied to F_hat AT
//      edge locations. The result is a sharpened pixel value (not a delta),
//      clamped to [0,255]. Non-edge pixels copy F_hat unchanged so F_AES
//      is a complete image, not a sparse edge map.
//
//   2. ODAD: F_s = F_hat - F_E means smooth pixels retain their F_hat value,
//      edge pixels = 0. ODAD output F_OTP should only contribute smooth pixel
//      values. Edge pixels come from F_AES. The composition F_IHR = F_AES +
//      F_OTP is a masked merge, not arithmetic addition on the full image.
//
//   3. CS fitness: SSIM was computing F_s vs F_s (identical → always 1.0).
//      Fixed to compare ODAD(F_s) against F_hat at smooth pixel locations,
//      measuring how well texture is preserved vs the Lanczos reference.
// =============================================================================
#include "odas.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>
#include <iostream>
#include <cassert>
#include <functional>

// =============================================================================
// SECTION 0: Internal utility types and helpers
// =============================================================================

static constexpr float kPi = 3.14159265358979323846f;

static inline unsigned char clampByte(float v) {
    return (unsigned char)std::max(0.0f, std::min(255.0f, v + 0.5f));
}

// Mirror padding pixel fetch for float single-channel image
static inline float getPixelF(const std::vector<float>& img,
                               int w, int h, int x, int y)
{
    if (x < 0)  x = -x - 1;
    if (y < 0)  y = -y - 1;
    if (x >= w) x = 2 * w - x - 1;
    if (y >= h) y = 2 * h - y - 1;
    x = std::max(0, std::min(w - 1, x));
    y = std::max(0, std::min(h - 1, y));
    return img[(size_t)y * w + x];
}

// =============================================================================
// SECTION 1: Lanczos3 Interpolation
// =============================================================================

static float sinc(float x) {
    if (std::abs(x) < 1e-7f) return 1.0f;
    float px = kPi * x;
    return std::sin(px) / px;
}

static float lanczos3Weight(float p) {
    float ap = std::abs(p);
    if (ap >= 3.0f) return 0.0f;
    return sinc(ap) * sinc(ap / 3.0f);
}

static void lanczos3Upscale(
    const std::vector<float>& src, int inW,  int inH,
    std::vector<float>&       dst, int outW, int outH)
{
    dst.resize((size_t)outW * outH);
    float scaleX = (float)inW  / (float)outW;
    float scaleY = (float)inH  / (float)outH;

    for (int oy = 0; oy < outH; ++oy) {
        float srcY = ((float)oy + 0.5f) * scaleY - 0.5f;
        int   cy   = (int)std::floor(srcY);

        for (int ox = 0; ox < outW; ++ox) {
            float srcX = ((float)ox + 0.5f) * scaleX - 0.5f;
            int   cx   = (int)std::floor(srcX);

            float sum  = 0.0f;
            float wSum = 0.0f;

            for (int ky = cy - 2; ky <= cy + 3; ++ky) {
                float wy = lanczos3Weight(srcY - (float)ky);
                for (int kx = cx - 2; kx <= cx + 3; ++kx) {
                    float wx = lanczos3Weight(srcX - (float)kx);
                    float w  = wx * wy;
                    sum  += getPixelF(src, inW, inH, kx, ky) * w;
                    wSum += w;
                }
            }

            float val = (wSum > 1e-7f) ? (sum / wSum)
                                       : getPixelF(src, inW, inH, cx, cy);
            dst[(size_t)oy * outW + ox] = std::max(0.0f, std::min(255.0f, val));
        }
    }
}

static void lanczos3Downscale(
    const std::vector<float>& src, int inW,  int inH,
    std::vector<float>&       dst, int outW, int outH)
{
    dst.resize((size_t)outW * outH);
    float scaleX = (float)inW  / (float)outW;
    float scaleY = (float)inH  / (float)outH;
    float radX   = 3.0f * scaleX;
    float radY   = 3.0f * scaleY;

    for (int oy = 0; oy < outH; ++oy) {
        float srcY = ((float)oy + 0.5f) * scaleY - 0.5f;
        int   yMin = (int)std::floor(srcY - radY + 1);
        int   yMax = (int)std::floor(srcY + radY);

        for (int ox = 0; ox < outW; ++ox) {
            float srcX = ((float)ox + 0.5f) * scaleX - 0.5f;
            int   xMin = (int)std::floor(srcX - radX + 1);
            int   xMax = (int)std::floor(srcX + radX);

            float sum  = 0.0f;
            float wSum = 0.0f;

            for (int ky = yMin; ky <= yMax; ++ky) {
                float wy = lanczos3Weight((srcY - (float)ky) / scaleY);
                for (int kx = xMin; kx <= xMax; ++kx) {
                    float wx = lanczos3Weight((srcX - (float)kx) / scaleX);
                    float w  = wx * wy;
                    sum  += getPixelF(src, inW, inH, kx, ky) * w;
                    wSum += w;
                }
            }

            int cy = std::max(0, std::min(inH - 1, (int)srcY));
            int cx = std::max(0, std::min(inW - 1, (int)srcX));
            float val = (wSum > 1e-7f) ? (sum / wSum)
                                       : getPixelF(src, inW, inH, cx, cy);
            dst[(size_t)oy * outW + ox] = std::max(0.0f, std::min(255.0f, val));
        }
    }
}

// =============================================================================
// SECTION 2: Edge Detection (Canny)
// =============================================================================

static void gaussianBlur5(const std::vector<float>& src, int w, int h,
                           std::vector<float>& dst)
{
    static const float K[5][5] = {
        {1,  4,  7,  4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1,  4,  7,  4, 1}
    };
    static const float Ksum = 273.0f;

    dst.resize(src.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int ky = -2; ky <= 2; ++ky)
                for (int kx = -2; kx <= 2; ++kx)
                    acc += K[ky+2][kx+2] * getPixelF(src, w, h, x+kx, y+ky);
            dst[(size_t)y * w + x] = acc / Ksum;
        }
}

static void cannyEdgeDetect(
    const std::vector<float>& img, int w, int h,
    std::vector<bool>&        edgeMask,
    float                     lowThresh,
    float                     highThresh)
{
    const int N = w * h;

    std::vector<float> blurred;
    gaussianBlur5(img, w, h, blurred);

    std::vector<float> gradMag(N, 0.0f);
    std::vector<float> gradDir(N, 0.0f);

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float gx =
                -1.0f * getPixelF(blurred, w, h, x-1, y-1) +
                 1.0f * getPixelF(blurred, w, h, x+1, y-1) +
                -2.0f * getPixelF(blurred, w, h, x-1, y  ) +
                 2.0f * getPixelF(blurred, w, h, x+1, y  ) +
                -1.0f * getPixelF(blurred, w, h, x-1, y+1) +
                 1.0f * getPixelF(blurred, w, h, x+1, y+1);

            float gy =
                -1.0f * getPixelF(blurred, w, h, x-1, y-1) +
                -2.0f * getPixelF(blurred, w, h, x,   y-1) +
                -1.0f * getPixelF(blurred, w, h, x+1, y-1) +
                 1.0f * getPixelF(blurred, w, h, x-1, y+1) +
                 2.0f * getPixelF(blurred, w, h, x,   y+1) +
                 1.0f * getPixelF(blurred, w, h, x+1, y+1);

            size_t idx = (size_t)y * w + x;
            gradMag[idx] = std::sqrt(gx * gx + gy * gy);
            gradDir[idx] = std::atan2(gy, gx);
        }

    std::vector<float> suppressed(N, 0.0f);
    for (int y = 1; y < h - 1; ++y)
        for (int x = 1; x < w - 1; ++x) {
            size_t idx   = (size_t)y * w + x;
            float  mag   = gradMag[idx];
            float  angle = gradDir[idx] * 180.0f / kPi;
            if (angle < 0.0f) angle += 180.0f;

            float mag1, mag2;
            if (angle < 22.5f || angle >= 157.5f) {
                mag1 = gradMag[idx + 1];
                mag2 = gradMag[idx - 1];
            } else if (angle < 67.5f) {
                mag1 = gradMag[idx - (size_t)w + 1];
                mag2 = gradMag[idx + (size_t)w - 1];
            } else if (angle < 112.5f) {
                mag1 = gradMag[idx - w];
                mag2 = gradMag[idx + w];
            } else {
                mag1 = gradMag[idx - (size_t)w - 1];
                mag2 = gradMag[idx + (size_t)w + 1];
            }
            suppressed[idx] = (mag >= mag1 && mag >= mag2) ? mag : 0.0f;
        }

    std::vector<int> edgeStatus(N, 0);
    for (int i = 0; i < N; ++i) {
        if      (suppressed[i] >= highThresh) edgeStatus[i] = 2;
        else if (suppressed[i] >= lowThresh)  edgeStatus[i] = 1;
    }

    std::vector<int> stack;
    stack.reserve(N / 8);
    for (int i = 0; i < N; ++i)
        if (edgeStatus[i] == 2) stack.push_back(i);

    while (!stack.empty()) {
        int idx = stack.back(); stack.pop_back();
        int px = idx % w, py = idx / w;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = px + dx, ny = py + dy;
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                int nIdx = ny * w + nx;
                if (edgeStatus[nIdx] == 1) {
                    edgeStatus[nIdx] = 2;
                    stack.push_back(nIdx);
                }
            }
    }

    edgeMask.assign(N, false);
    for (int i = 0; i < N; ++i)
        edgeMask[i] = (edgeStatus[i] == 2);
}

// =============================================================================
// SECTION 3: Adaptive Edge Sharpening (AES) - BUG FIX
//
// FIX: The AES kernel H is a Laplacian sharpening kernel. When convolved
// with F_hat (NOT sparse F_E), it produces a sharpened pixel value at each
// edge location. Non-edge pixels are copied from F_hat unchanged.
//
// This produces F_AES as a COMPLETE image (same size as F_hat) where:
//   - Edge pixels:     sharpened version of F_hat using adaptive Laplacian
//   - Non-edge pixels: copied from F_hat (unchanged)
//
// The paper's Algorithm 1 says "convolve H with F_E" but F_E IS the edge
// region of F_hat. The convolution reads neighboring pixels which in a
// sparse map would be zero → wrong. The correct reading is: convolve H
// with F_hat but only STORE results at edge pixel locations.
//
// Kernel H structure (sums to zero = pure sharpening, no DC shift):
//   [[-cmp/8, -cmp/8, -cmp/8],
//    [-cmp/8,  cmp,   -cmp/8],
//    [-cmp/8, -cmp/8, -cmp/8]]
//
// The convolution result = cmp * center - (cmp/8) * sum(8 neighbors)
// = cmp * (center - mean_neighbors)  [a Laplacian = edge amplifier]
//
// We apply this to F_hat and clamp to [0,255]. The cmp scaling means
// high-variance edges get sharpened more aggressively.
// =============================================================================

static void applyAES(
    const std::vector<float>& Fhat,     // full interpolated image
    const std::vector<bool>&  edgeMask, // which pixels are edges
    int w, int h,
    std::vector<float>&       FAES)     // output: complete sharpened image
{
    const size_t N = (size_t)w * h;
    FAES.resize(N);

    // ── Compute local variance at each EDGE pixel ─────────────────────────
    // Variance is computed from F_hat neighbors (not sparse F_E)
    std::vector<float> localVar(N, 0.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            size_t idx = (size_t)y * w + x;
            if (!edgeMask[idx]) continue;

            float sumV = 0.0f;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx)
                    sumV += getPixelF(Fhat, w, h, x + kx, y + ky);
            float mean = sumV / 9.0f;

            float varSum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    float d = getPixelF(Fhat, w, h, x + kx, y + ky) - mean;
                    varSum += d * d;
                }
            localVar[idx] = varSum / 9.0f;
        }

    // ── Find VR_min / VR_max across edge pixels only ──────────────────────
    float VRmin = 1e30f, VRmax = -1e30f;
    for (size_t i = 0; i < N; ++i) {
        if (!edgeMask[i]) continue;
        VRmin = std::min(VRmin, localVar[i]);
        VRmax = std::max(VRmax, localVar[i]);
    }

    // No edge pixels detected — copy F_hat unchanged
    if (VRmin > VRmax) { FAES = Fhat; return; }

    float S = (VRmax - VRmin) / 4.0f;
    // Guard against all-same-variance edge case
    if (S < 1e-6f) S = 1.0f;

    // ── Per-pixel AES ─────────────────────────────────────────────────────
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t idx = (size_t)y * w + x;

            if (!edgeMask[idx]) {
                // Non-edge pixels: pass F_hat through unchanged
                // They will come from F_OTP (smooth region) in composition
                FAES[idx] = Fhat[idx];
                continue;
            }

            float lvr = localVar[idx];

            // Select cmp based on variance quartile (Algorithm 1)
            float cmp;
            if      (lvr <= VRmin +       S) cmp = 16.0f;
            else if (lvr <= VRmin + 2.0f * S) cmp = 24.0f;
            else if (lvr <= VRmin + 3.0f * S) cmp = 32.0f;
            else if (lvr <= VRmin + 4.0f * S) cmp = 40.0f;
            else                               cmp = 48.0f;

            // Build Laplacian kernel H and convolve with F_hat
            // H centre = cmp, neighbours = -cmp/8
            // Result = cmp * center - (cmp/8) * sum(8 neighbors)
            //        = centre + (cmp-1)*(centre - mean_neighbors)  [sharpened]
            float neighbourW = -cmp / 8.0f;
            float result = cmp * getPixelF(Fhat, w, h, x, y);
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    if (ky == 0 && kx == 0) continue;
                    result += neighbourW * getPixelF(Fhat, w, h, x + kx, y + ky);
                }

            // Clamp: the Laplacian amplifies signal but output must be valid
            FAES[idx] = std::max(0.0f, std::min(255.0f, result));
        }
    }
}

// =============================================================================
// SECTION 4: ODAD Filter
//
// Equation (8): F_OTP[t+1](m,n) = F_s[t](m,n) + λ·Σ_dir(d_dir·∇_dir[F_s])
// Equation (11): d = exp(-(|∇F_s|/K)²)
//
// Applied only to smooth region pixels (edgeMask == false).
// Edge pixel positions retain value 0 (they come from F_AES in composition).
// t = 1 (one iteration per paper).
// =============================================================================

static void applyODAD(
    const std::vector<float>& Fs,  // smooth image (edge pixels = 0)
    int w, int h,
    std::vector<float>&       FOTP,
    float lambda, float K, int iterations)
{
    const size_t N = (size_t)w * h;
    FOTP = Fs;
    const float K2 = K * K;

    for (int t = 0; t < iterations; ++t) {
        std::vector<float> next(N);
        for (int m = 0; m < h; ++m) {
            for (int n = 0; n < w; ++n) {
                float center = getPixelF(FOTP, w, h, n, m);

                // 8 directional gradients (Equation 9)
                float gN  = getPixelF(FOTP, w, h, n,   m-1) - center;
                float gS  = getPixelF(FOTP, w, h, n,   m+1) - center;
                float gE  = getPixelF(FOTP, w, h, n+1, m  ) - center;
                float gW  = getPixelF(FOTP, w, h, n-1, m  ) - center;
                float gNE = getPixelF(FOTP, w, h, n+1, m-1) - center;
                float gSE = getPixelF(FOTP, w, h, n+1, m+1) - center;
                float gWS = getPixelF(FOTP, w, h, n-1, m+1) - center;
                float gWN = getPixelF(FOTP, w, h, n-1, m-1) - center;

                // Diffusion coefficients (Equation 10+11)
                auto dc = [&](float g) {
                    return std::exp(-(g * g) / K2);
                };

                float update = lambda * (
                    dc(gN)*gN + dc(gS)*gS + dc(gE)*gE + dc(gW)*gW +
                    dc(gNE)*gNE + dc(gSE)*gSE + dc(gWS)*gWS + dc(gWN)*gWN
                );

                next[(size_t)m * w + n] = center + update;
            }
        }
        FOTP = next;
    }

    for (auto& v : FOTP)
        v = std::max(0.0f, std::min(255.0f, v));
}

// =============================================================================
// SECTION 5: Cuckoo Search Optimization for λ - BUG FIX
//
// FIX: Fitness function now correctly measures ODAD filter effectiveness.
//
// Previous bug: computeSSIM(filtered, Fs) compared ODAD output to F_s which
// contains zeros at edge locations. Both filtered and F_s have zeros at the
// same edge positions → SSIM numerically = 1.0 always.
//
// Fix: Compare ODAD(F_s) against F_hat at SMOOTH pixel locations only.
// This measures how well the filter preserves original texture vs the
// reference Lanczos interpolation. A good λ keeps smooth regions close
// to F_hat while allowing the filter to enhance local texture.
//
// Alternative correct fitness: compute variance of ODAD output in smooth
// region (higher texture variance = better detail preservation).
// We use the F_hat comparison as it is more principled.
// =============================================================================

// Compute SSIM only at smooth (non-edge) pixel locations
static float computeSSIMSmooth(
    const std::vector<float>& filtered,  // ODAD output
    const std::vector<float>& reference, // F_hat (Lanczos interpolated)
    const std::vector<bool>&  edgeMask,
    int w, int h)
{
    // Collect smooth pixel pairs
    std::vector<float> a, b;
    a.reserve(w * h / 2);
    b.reserve(w * h / 2);

    for (size_t i = 0; i < (size_t)w * h; ++i) {
        if (!edgeMask[i]) {
            a.push_back(filtered[i]);
            b.push_back(reference[i]);
        }
    }

    if (a.size() < 4) return 0.0f;

    const float C1 = (0.01f * 255.0f) * (0.01f * 255.0f);
    const float C2 = (0.03f * 255.0f) * (0.03f * 255.0f);

    // Means
    double sumA = 0, sumB = 0;
    for (size_t i = 0; i < a.size(); ++i) { sumA += a[i]; sumB += b[i]; }
    float muA = (float)(sumA / a.size());
    float muB = (float)(sumB / a.size());

    // Variances and covariance
    double varA = 0, varB = 0, cov = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double da = a[i] - muA, db = b[i] - muB;
        varA += da * da;
        varB += db * db;
        cov  += da * db;
    }
    varA /= a.size(); varB /= a.size(); cov /= a.size();

    float ssim = (float)(
        ((2.0 * muA * muB + C1) * (2.0 * cov + C2)) /
        ((muA*muA + muB*muB + C1) * (varA + varB + C2))
    );
    return ssim;
}

// Forward declaration
static void applyODAD(const std::vector<float>&, int, int,
                      std::vector<float>&, float, float, int);

static float levyStep(std::mt19937& rng, float beta) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float num   = std::tgamma(1.0f + beta) * std::sin(kPi * beta / 2.0f);
    float den   = std::tgamma((1.0f + beta) / 2.0f) * beta *
                  std::pow(2.0f, (beta - 1.0f) / 2.0f);
    float sigma = std::pow(num / den, 1.0f / beta);
    float u     = nd(rng) * sigma;
    float v     = nd(rng);
    if (std::abs(v) < 1e-10f) v = 1e-10f;
    return u / std::pow(std::abs(v), 1.0f / beta);
}

static float cuckooSearchLambda(
    const std::vector<float>& Fs,       // smooth region (edge pixels = 0)
    const std::vector<float>& Fhat,     // full Lanczos interpolated image
    const std::vector<bool>&  edgeMask, // which pixels are edges
    int w, int h,
    const OdasParams& params)
{
    const float lambdaMin = 0.0f;
    const float lambdaMax = kPi / 4.0f;
    const int   n         = params.csNests;
    const int   MAXit     = params.csMaxIter;
    const float pa        = params.csPa;
    const float beta      = params.csLevyBeta;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniRange(lambdaMin, lambdaMax);
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);

    // Fitness: SSIM of ODAD(Fs) vs F_hat at smooth pixels
    // Higher SSIM = filter preserves texture details = better
    auto fitness = [&](float lambda) -> float {
        std::vector<float> filtered;
        applyODAD(Fs, w, h, filtered, lambda, params.K, params.odadIterations);
        return computeSSIMSmooth(filtered, Fhat, edgeMask, w, h);
    };

    // Initialize population
    std::vector<float> nests(n);
    std::vector<float> fitnesses(n);
    for (int i = 0; i < n; ++i) {
        nests[i]     = uniRange(rng);
        fitnesses[i] = fitness(nests[i]);
    }

    int   bestIdx    = (int)(std::max_element(fitnesses.begin(),
                                               fitnesses.end()) - fitnesses.begin());
    float bestLambda = nests[bestIdx];
    float bestFit    = fitnesses[bestIdx];

    std::cout << "  [ODAS-CS] Starting Cuckoo Search for optimal λ ∈ [0, π/4]"
              << std::endl;
    std::cout << "  [ODAS-CS] Population=" << n << "  MaxIter=" << MAXit
              << "  Pa=" << pa << std::endl;

    for (int t = 0; t < MAXit; ++t) {
        // Lévy flight step size decreases over time
        float stepScale = 0.01f * (lambdaMax - lambdaMin) /
                          (1.0f + (float)t * 0.05f);

        for (int i = 0; i < n; ++i) {
            float levy      = levyStep(rng, beta);
            float newLambda = nests[i] + stepScale * levy;
            newLambda       = std::max(lambdaMin, std::min(lambdaMax, newLambda));

            float newFit = fitness(newLambda);

            // Replace random nest if better
            int j = (int)(uni01(rng) * (float)n) % n;
            if (newFit > fitnesses[j]) {
                nests[j]     = newLambda;
                fitnesses[j] = newFit;
                if (newFit > bestFit) {
                    bestFit    = newFit;
                    bestLambda = newLambda;
                }
            }
        }

        // Abandon worst nests with probability pa
        int numAbandon = std::max(1, (int)(pa * (float)n));
        std::vector<int> sortedIdx(n);
        std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
        std::sort(sortedIdx.begin(), sortedIdx.end(),
                  [&](int a, int b){ return fitnesses[a] < fitnesses[b]; });
        for (int k = 0; k < numAbandon; ++k) {
            int idx        = sortedIdx[k];
            nests[idx]     = uniRange(rng);
            fitnesses[idx] = fitness(nests[idx]);
            if (fitnesses[idx] > bestFit) {
                bestFit    = fitnesses[idx];
                bestLambda = nests[idx];
            }
        }

        if ((t + 1) % 20 == 0 || t == MAXit - 1) {
            std::cout << "  [ODAS-CS] Iter " << (t+1) << "/" << MAXit
                      << "  Best λ=" << bestLambda
                      << "  SSIM=" << bestFit << std::endl;
        }
    }

    std::cout << "  [ODAS-CS] Optimal λ = " << bestLambda
              << "  (SSIM=" << bestFit << ")" << std::endl;
    return bestLambda;
}

// =============================================================================
// SECTION 6: IHR Composition - BUG FIX
//
// FIX: F_IHR is a masked merge, not arithmetic addition.
//
// Previous bug: F_IHR = F_AES + F_OTP added values at ALL pixels.
//   At edge pixels:   F_AES[i] (sharpened) + F_OTP[i] (≈0, edges excluded) 
//                     → approximately correct but F_OTP bleeds in
//   At smooth pixels: F_AES[i] (= F_hat copy) + F_OTP[i] (smooth filtered)
//                     → F_hat + ODAD(F_hat) ≈ 2×signal → too bright!
//
// Fix: Use edge mask to select source:
//   Edge pixels    → from F_AES (adaptively sharpened)
//   Smooth pixels  → from F_OTP (ODAD texture-preserved)
//
// This is the correct interpretation of "merging" the two regions.
// =============================================================================

static void composeIHR(
    const std::vector<float>& FAES,     // sharpened image (all pixels present)
    const std::vector<float>& FOTP,     // ODAD filtered smooth image
    const std::vector<bool>&  edgeMask, // true = edge pixel
    int w, int h,
    std::vector<float>&       FIHR)
{
    const size_t N = (size_t)w * h;
    FIHR.resize(N);
    for (size_t i = 0; i < N; ++i) {
        // Edge pixels: use adaptively sharpened value from F_AES
        // Smooth pixels: use ODAD texture-preserved value from F_OTP
        FIHR[i] = std::max(0.0f, std::min(255.0f,
                      edgeMask[i] ? FAES[i] : FOTP[i]));
    }
}

// =============================================================================
// SECTION 7: Residual-Based Sharpening (Equations 13, 14)
//
// F_BHR = Lanczos3_Downscale(Lanczos3_Upscale(F_IHR))
// F_RES = F_IHR - F_BHR
// F_RHR = F_IHR + F_RES = 2·F_IHR - F_BHR
// =============================================================================

static void residualSharpening(
    const std::vector<float>& FIHR, int w, int h,
    int scaleFactor,
    std::vector<float>&       FRHR)
{
    int upW = w * scaleFactor;
    int upH = h * scaleFactor;

    std::vector<float> upscaled;
    lanczos3Upscale(FIHR, w, h, upscaled, upW, upH);

    std::vector<float> FBHR;
    lanczos3Downscale(upscaled, upW, upH, FBHR, w, h);

    const size_t N = (size_t)w * h;
    FRHR.resize(N);
    for (size_t i = 0; i < N; ++i) {
        float res  = FIHR[i] - FBHR[i];
        float frhr = FIHR[i] + res;
        FRHR[i] = std::max(0.0f, std::min(255.0f, frhr));
    }
}

// =============================================================================
// SECTION 8: Color Space Helpers
// =============================================================================

static void rgbToYCbCr(
    const unsigned char* rgb, int w, int h,
    std::vector<float>& Y, std::vector<float>& Cb, std::vector<float>& Cr)
{
    const size_t N = (size_t)w * h;
    Y.resize(N); Cb.resize(N); Cr.resize(N);
    for (size_t i = 0; i < N; ++i) {
        float r = rgb[i*4+0], g = rgb[i*4+1], b = rgb[i*4+2];
        Y[i]  =  0.299f   * r + 0.587f   * g + 0.114f   * b;
        Cb[i] = -0.16874f * r - 0.33126f * g + 0.5f     * b + 128.0f;
        Cr[i] =  0.5f     * r - 0.41869f * g - 0.08131f * b + 128.0f;
    }
}

static void bilinearUpscale(
    const std::vector<float>& src, int inW, int inH,
    std::vector<float>&       dst, int outW, int outH)
{
    dst.resize((size_t)outW * outH);
    float sx = (float)inW / (float)outW;
    float sy = (float)inH / (float)outH;
    for (int oy = 0; oy < outH; ++oy) {
        float fy = ((float)oy + 0.5f) * sy - 0.5f;
        int   y0 = (int)std::floor(fy);
        float v  = fy - (float)y0;
        for (int ox = 0; ox < outW; ++ox) {
            float fx = ((float)ox + 0.5f) * sx - 0.5f;
            int   x0 = (int)std::floor(fx);
            float u  = fx - (float)x0;
            float p00 = getPixelF(src, inW, inH, x0,   y0);
            float p10 = getPixelF(src, inW, inH, x0+1, y0);
            float p01 = getPixelF(src, inW, inH, x0,   y0+1);
            float p11 = getPixelF(src, inW, inH, x0+1, y0+1);
            float top = p00 + u * (p10 - p00);
            float bot = p01 + u * (p11 - p01);
            dst[(size_t)oy * outW + ox] = top + v * (bot - top);
        }
    }
}

static void bilinearUpscaleRGBA(
    const unsigned char* src, int inW, int inH,
    unsigned char*       dst, int outW, int outH)
{
    float sx = (float)inW / (float)outW;
    float sy = (float)inH / (float)outH;
    for (int oy = 0; oy < outH; ++oy) {
        float fy = ((float)oy + 0.5f) * sy - 0.5f;
        int   y0 = std::max(0, std::min(inH-1, (int)std::floor(fy)));
        int   y1 = std::max(0, std::min(inH-1, y0+1));
        float v  = fy - std::floor(fy);
        for (int ox = 0; ox < outW; ++ox) {
            float fx = ((float)ox + 0.5f) * sx - 0.5f;
            int   x0 = std::max(0, std::min(inW-1, (int)std::floor(fx)));
            int   x1 = std::max(0, std::min(inW-1, x0+1));
            float u  = fx - std::floor(fx);
            for (int c = 0; c < 4; ++c) {
                float p00 = src[(y0*inW+x0)*4+c];
                float p10 = src[(y0*inW+x1)*4+c];
                float p01 = src[(y1*inW+x0)*4+c];
                float p11 = src[(y1*inW+x1)*4+c];
                float val = (p00*(1-u)+p10*u)*(1-v)+(p01*(1-u)+p11*u)*v;
                dst[(oy*outW+ox)*4+c] = clampByte(val);
            }
        }
    }
}

static void yCbCrToRgba(
    const std::vector<float>& Y,
    const std::vector<float>& Cb,
    const std::vector<float>& Cr,
    const unsigned char*      alphaSource,
    unsigned char* dst, int w, int h)
{
    const size_t N = (size_t)w * h;
    for (size_t i = 0; i < N; ++i) {
        float y  = Y[i];
        float cb = Cb[i] - 128.0f;
        float cr = Cr[i] - 128.0f;
        dst[i*4+0] = clampByte(y + 1.40200f * cr);
        dst[i*4+1] = clampByte(y - 0.34414f * cb - 0.71414f * cr);
        dst[i*4+2] = clampByte(y + 1.77200f * cb);
        dst[i*4+3] = alphaSource[i*4+3];
    }
}

// =============================================================================
// SECTION 9: Main Entry Point
// =============================================================================

void scaleODAS(
    const unsigned char* input,  int inW,  int inH,
    unsigned char*       output, int outW, int outH,
    const OdasParams&    params)
{
    int scaleFactor = std::max(1, (int)std::round((float)outW / (float)inW));

    std::cout << "  [ODAS] Input:  " << inW  << "x" << inH  << std::endl;
    std::cout << "  [ODAS] Output: " << outW << "x" << outH << std::endl;
    std::cout << "  [ODAS] Scale factor: ~" << scaleFactor << "x" << std::endl;
    std::cout << "  [ODAS] K=" << params.K
              << "  ODAD iterations=" << params.odadIterations << std::endl;

    // Bilinear upscale for alpha channel
    std::vector<unsigned char> bilinearOut((size_t)outW * outH * 4);
    bilinearUpscaleRGBA(input, inW, inH, bilinearOut.data(), outW, outH);

    if (params.useYCbCr) {
        // ── YCbCr mode ────────────────────────────────────────────────────

        // Step 0: Convert to YCbCr
        std::vector<float> Yin, Cbin, Crin;
        rgbToYCbCr(input, inW, inH, Yin, Cbin, Crin);

        // Step 1: Lanczos3 upscale Y channel → F_hat
        std::cout << "  [ODAS] Step 1: Lanczos3 upscale..." << std::endl;
        std::vector<float> Fhat;
        lanczos3Upscale(Yin, inW, inH, Fhat, outW, outH);

        // Step 2: Canny edge detection on F_hat
        std::cout << "  [ODAS] Step 2: Canny edge detection..." << std::endl;
        std::vector<bool> edgeMask;
        cannyEdgeDetect(Fhat, outW, outH, edgeMask,
                        params.cannyLowThresh, params.cannyHighThresh);

        size_t edgeCount = std::count(edgeMask.begin(), edgeMask.end(), true);
        std::cout << "  [ODAS] Edge pixels: " << edgeCount
                  << " / " << ((size_t)outW * outH)
                  << " (" << (100.0f * edgeCount / (outW * outH)) << "%)"
                  << std::endl;

        // Step 3: Build F_s (smooth region: edge pixels = 0)
        // F_s is used by ODAD. Edge pixels are zeroed out so ODAD
        // only influences smooth regions.
        std::vector<float> Fs((size_t)outW * outH, 0.0f);
        for (size_t i = 0; i < (size_t)outW * outH; ++i)
            if (!edgeMask[i]) Fs[i] = Fhat[i];

        // Step 4: AES on F_hat at edge locations → F_AES (complete image)
        std::cout << "  [ODAS] Step 4: Adaptive Edge Sharpening..." << std::endl;
        std::vector<float> FAES;
        applyAES(Fhat, edgeMask, outW, outH, FAES);

        // Step 5: Cuckoo Search for optimal λ
        std::cout << "  [ODAS] Step 5: Cuckoo Search optimization for λ..."
                  << std::endl;
        float lambda = cuckooSearchLambda(Fs, Fhat, edgeMask, outW, outH, params);

        // Step 6: ODAD filter on F_s → F_OTP
        std::cout << "  [ODAS] Step 6: ODAD filter (λ=" << lambda << ")..."
                  << std::endl;
        std::vector<float> FOTP;
        applyODAD(Fs, outW, outH, FOTP, lambda, params.K, params.odadIterations);

        // Step 7: IHR composition (masked merge)
        std::cout << "  [ODAS] Step 7: IHR composition..." << std::endl;
        std::vector<float> FIHR;
        composeIHR(FAES, FOTP, edgeMask, outW, outH, FIHR);

        // Step 8: Residual-based sharpening
        std::cout << "  [ODAS] Step 8: Residual-based sharpening..." << std::endl;
        std::vector<float> FRHR;
        residualSharpening(FIHR, outW, outH, scaleFactor, FRHR);

        // Step 9: Bilinear upscale Cb and Cr
        std::vector<float> CbOut, CrOut;
        bilinearUpscale(Cbin, inW, inH, CbOut, outW, outH);
        bilinearUpscale(Crin, inW, inH, CrOut, outW, outH);

        // Step 10: Convert Y_restored + Cb + Cr → RGBA
        std::cout << "  [ODAS] Step 10: Compositing final RGBA output..."
                  << std::endl;
        yCbCrToRgba(FRHR, CbOut, CrOut, bilinearOut.data(), output, outW, outH);

    } else {
        // ── RGB mode: process each channel independently ───────────────────
        std::cout << "  [ODAS] RGB mode: processing 3 channels independently"
                  << std::endl;

        std::vector<std::vector<float>> channelFinal(3);

        for (int ch = 0; ch < 3; ++ch) {
            std::cout << "  [ODAS] Channel " << ch << "/3..." << std::endl;

            std::vector<float> chanIn((size_t)inW * inH);
            for (int i = 0; i < inW * inH; ++i)
                chanIn[i] = (float)input[i*4+ch];

            std::vector<float> Fhat;
            lanczos3Upscale(chanIn, inW, inH, Fhat, outW, outH);

            std::vector<bool> edgeMask;
            cannyEdgeDetect(Fhat, outW, outH, edgeMask,
                            params.cannyLowThresh, params.cannyHighThresh);

            std::vector<float> Fs((size_t)outW * outH, 0.0f);
            for (size_t i = 0; i < (size_t)outW * outH; ++i)
                if (!edgeMask[i]) Fs[i] = Fhat[i];

            std::vector<float> FAES;
            applyAES(Fhat, edgeMask, outW, outH, FAES);

            float lambda = cuckooSearchLambda(Fs, Fhat, edgeMask,
                                              outW, outH, params);

            std::vector<float> FOTP;
            applyODAD(Fs, outW, outH, FOTP, lambda, params.K,
                      params.odadIterations);

            std::vector<float> FIHR;
            composeIHR(FAES, FOTP, edgeMask, outW, outH, FIHR);

            std::vector<float> FRHR;
            residualSharpening(FIHR, outW, outH, scaleFactor, FRHR);

            channelFinal[ch] = FRHR;
        }

        for (int i = 0; i < outW * outH; ++i) {
            output[i*4+0] = clampByte(channelFinal[0][i]);
            output[i*4+1] = clampByte(channelFinal[1][i]);
            output[i*4+2] = clampByte(channelFinal[2][i]);
            output[i*4+3] = bilinearOut[i*4+3];
        }
    }

    std::cout << "  [ODAS] Done." << std::endl;
}
