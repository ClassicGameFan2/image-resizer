// =============================================================================
// odas.cpp — v7: Faithful implementation matching Panda & Meher 2024
//
// PIPELINE (matches paper Section 4 exactly):
//   F_hat  = Lanczos3_Upscale(F_LR)                     [Eq. 3]
//   F_E    = EdgeDetect(F_hat)       -- actual pixel values at edges
//   F_s    = F_hat - F_E             -- smooth region    [Eq. 7]
//   F_AES  = Convolve(H_adaptive, F_E)                   [Eq. 5, Alg 1]
//   F_OTP  = ODAD(F_s, lambda, K)    -- 1 iteration      [Eq. 8-11]
//   F_IHR  = F_AES + F_OTP          -- additive          [Eq. 12]
//   F_BHR  = L3dn(L3up(F_IHR))      -- blur reference    
//   F_RES  = F_IHR - F_BHR                               [Eq. 13]
//   F_RHR  = F_IHR + F_RES = 2*F_IHR - F_BHR            [Eq. 14]
//
// KEY FIXES vs v6:
//   1. F_E contains ACTUAL pixel values (not a mask). F_s = Fhat - F_E.
//   2. AES: convolve H with F_E directly (not unsharp mask on Fhat).
//   3. IHR: F_AES + F_OTP (additive, not selector).
//   4. ODAD: smooth region only (F_s), edge pixels are fixed boundaries.
//   5. CS fitness: SSIM of (F_OTP vs F_s) — maximize structural similarity.
//      λ=0 gives SSIM=1.0 trivially, so we maximize SSIM while requiring
//      λ > 0 (we want the largest λ that keeps SSIM high).
//      Correct formulation: maximize λ subject to SSIM(F_OTP, F_s) > threshold.
//   6. Levy flight: proper step without decay collapse.
// =============================================================================
#include "odas.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>

static constexpr float kPi = 3.14159265358979323846f;

static inline unsigned char clampByte(float v) {
    return (unsigned char)std::max(0.0f, std::min(255.0f, v + 0.5f));
}

// Mirror-reflect boundary for float image
static inline float getPixelF(const std::vector<float>& img,
    int w, int h, int x, int y)
{
    if (x < 0)  x = -x - 1;
    if (y < 0)  y = -y - 1;
    if (x >= w) x = 2*w - x - 1;
    if (y >= h) y = 2*h - y - 1;
    x = std::max(0, std::min(w-1, x));
    y = std::max(0, std::min(h-1, y));
    return img[(size_t)y*w + x];
}

// =============================================================================
// LANCZOS3 UPSCALE / DOWNSCALE
// =============================================================================
static float sincF(float x) {
    if (std::abs(x) < 1e-7f) return 1.0f;
    float px = kPi * x;
    return std::sin(px) / px;
}

static float L3w(float p) {
    float a = std::abs(p);
    return (a < 3.0f) ? sincF(a) * sincF(a / 3.0f) : 0.0f;
}

static void L3up(const std::vector<float>& src, int iw, int ih,
    std::vector<float>& dst, int ow, int oh)
{
    dst.resize((size_t)ow * oh);
    float sx = (float)iw / ow, sy = (float)ih / oh;
    for (int oy = 0; oy < oh; ++oy) {
        float srcY = ((float)oy + 0.5f) * sy - 0.5f;
        int cy = (int)std::floor(srcY);
        for (int ox = 0; ox < ow; ++ox) {
            float srcX = ((float)ox + 0.5f) * sx - 0.5f;
            int cx = (int)std::floor(srcX);
            float s = 0, ws = 0;
            for (int ky = cy-2; ky <= cy+3; ++ky) {
                float wy = L3w(srcY - ky);
                for (int kx = cx-2; kx <= cx+3; ++kx) {
                    float w = L3w(srcX - kx) * wy;
                    s  += getPixelF(src, iw, ih, kx, ky) * w;
                    ws += w;
                }
            }
            dst[(size_t)oy*ow + ox] = std::max(0.0f, std::min(255.0f,
                (ws > 1e-7f) ? s/ws : getPixelF(src, iw, ih, cx, cy)));
        }
    }
}

static void L3dn(const std::vector<float>& src, int iw, int ih,
    std::vector<float>& dst, int ow, int oh)
{
    dst.resize((size_t)ow * oh);
    float sx = (float)iw / ow, sy = (float)ih / oh;
    float rx = 3.0f * sx, ry = 3.0f * sy;
    for (int oy = 0; oy < oh; ++oy) {
        float srcY = ((float)oy + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(srcY - ry + 1);
        int y1 = (int)std::floor(srcY + ry);
        for (int ox = 0; ox < ow; ++ox) {
            float srcX = ((float)ox + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(srcX - rx + 1);
            int x1 = (int)std::floor(srcX + rx);
            float s = 0, ws = 0;
            for (int ky = y0; ky <= y1; ++ky) {
                float wy = L3w((srcY - ky) / sy);
                for (int kx = x0; kx <= x1; ++kx) {
                    float w = L3w((srcX - kx) / sx) * wy;
                    s  += getPixelF(src, iw, ih, kx, ky) * w;
                    ws += w;
                }
            }
            int cy = std::max(0, std::min(ih-1, (int)srcY));
            int cx = std::max(0, std::min(iw-1, (int)srcX));
            dst[(size_t)oy*ow + ox] = std::max(0.0f, std::min(255.0f,
                (ws > 1e-7f) ? s/ws : getPixelF(src, iw, ih, cx, cy)));
        }
    }
}

// =============================================================================
// CANNY EDGE DETECTION
// Returns: edgeMask (bool), F_E (actual pixel values at edge locations),
//          F_s (Fhat minus edge pixels = smooth region)
// Paper Eq. 7: F_s = F_hat - F_E
// =============================================================================
static void gblur5(const std::vector<float>& src, int w, int h,
    std::vector<float>& dst)
{
    // 5x5 Gaussian kernel (sum=273)
    static const float K[5][5] = {
        {1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1}
    };
    dst.resize(src.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0;
            for (int ky = -2; ky <= 2; ++ky)
                for (int kx = -2; kx <= 2; ++kx)
                    a += K[ky+2][kx+2] * getPixelF(src, w, h, x+kx, y+ky);
            dst[(size_t)y*w + x] = a / 273.0f;
        }
}

// Builds edgeMask, F_E (pixel values at edges, 0 elsewhere),
// and F_s (pixel values at smooth pixels, 0 at edges).
static void cannyAndSplit(const std::vector<float>& Fhat,
    int w, int h,
    std::vector<bool>&  edgeMask,  // true = edge pixel
    std::vector<float>& FE,        // F_E: actual values at edges
    std::vector<float>& Fs,        // F_s: actual values at smooth pixels
    float lo, float hi)
{
    int N = w * h;
    std::vector<float> bl;
    gblur5(Fhat, w, h, bl);

    std::vector<float> gm(N, 0), gd(N, 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float gx =
                -getPixelF(bl,w,h,x-1,y-1) + getPixelF(bl,w,h,x+1,y-1)
                -2*getPixelF(bl,w,h,x-1,y) +2*getPixelF(bl,w,h,x+1,y)
                -getPixelF(bl,w,h,x-1,y+1) + getPixelF(bl,w,h,x+1,y+1);
            float gy =
                -getPixelF(bl,w,h,x-1,y-1) -2*getPixelF(bl,w,h,x,y-1)
                -getPixelF(bl,w,h,x+1,y-1) + getPixelF(bl,w,h,x-1,y+1)
                +2*getPixelF(bl,w,h,x,y+1) + getPixelF(bl,w,h,x+1,y+1);
            size_t i = (size_t)y*w + x;
            gm[i] = std::sqrt(gx*gx + gy*gy);
            gd[i] = std::atan2(gy, gx);
        }

    // Non-maximum suppression
    std::vector<float> sup(N, 0);
    for (int y = 1; y < h-1; ++y)
        for (int x = 1; x < w-1; ++x) {
            size_t i = (size_t)y*w + x;
            float m = gm[i];
            float a = gd[i] * 180.0f / kPi;
            if (a < 0) a += 180.0f;
            float m1, m2;
            if      (a < 22.5f  || a >= 157.5f) { m1 = gm[i+1];            m2 = gm[i-1]; }
            else if (a < 67.5f)                  { m1 = gm[i-(size_t)w+1]; m2 = gm[i+(size_t)w-1]; }
            else if (a < 112.5f)                 { m1 = gm[i-(size_t)w];   m2 = gm[i+(size_t)w]; }
            else                                 { m1 = gm[i-(size_t)w-1]; m2 = gm[i+(size_t)w+1]; }
            sup[i] = (m >= m1 && m >= m2) ? m : 0;
        }

    // Hysteresis thresholding
    std::vector<int> st(N, 0);
    for (int i = 0; i < N; ++i) {
        if      (sup[i] >= hi) st[i] = 2;
        else if (sup[i] >= lo) st[i] = 1;
    }
    std::vector<int> stk;
    stk.reserve(N / 8);
    for (int i = 0; i < N; ++i)
        if (st[i] == 2) stk.push_back(i);
    while (!stk.empty()) {
        int idx = stk.back(); stk.pop_back();
        int px = idx % w, py = idx / w;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (!dx && !dy) continue;
                int nx = px+dx, ny = py+dy;
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                int ni = ny*w + nx;
                if (st[ni] == 1) { st[ni] = 2; stk.push_back(ni); }
            }
    }

    // Build edgeMask, F_E, F_s
    // Paper Eq. 7: F_s = F_hat - F_E
    // F_E[i] = Fhat[i] if edge, else 0
    // F_s[i] = 0 if edge, else Fhat[i]  (= Fhat - F_E)
    edgeMask.assign(N, false);
    FE.assign(N, 0.0f);
    Fs.assign(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        if (st[i] == 2) {
            edgeMask[i] = true;
            FE[i] = Fhat[i];   // edge: F_E gets the pixel value
            Fs[i] = 0.0f;      // smooth: 0 at edge locations
        } else {
            edgeMask[i] = false;
            FE[i] = 0.0f;      // edge: 0 at smooth locations
            Fs[i] = Fhat[i];   // smooth: gets the pixel value
        }
    }
}

// =============================================================================
// AES — Adaptive Edge Sharpening (Algorithm 1, paper Section 4.2)
//
// Convolves F_E with adaptive 3x3 Laplacian kernel H.
// H has center = cmp, neighbors = -cmp/8 (sum = 0).
// cmp is chosen based on local variance of F_E in 5 levels: {16,24,32,40,48}
//
// F_AES(m,n) = sum_{k=-1}^{1} sum_{l=-1}^{1} H(k,l) * F_E(m+k, n+l)
//
// Note: F_E has actual pixel values at edge locations, 0 elsewhere.
// The convolution produces the sharpened edge image.
// =============================================================================
static void applyAES(const std::vector<float>& FE,
    const std::vector<bool>& edgeMask,
    int w, int h,
    std::vector<float>& FAES)
{
    size_t N = (size_t)w * h;
    FAES.resize(N, 0.0f);

    // Step 1: Compute local variance for all edge pixels (3x3 window of F_E)
    std::vector<float> lv(N, 0.0f);
    float VRmin = 1e30f, VRmax = -1e30f;

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            size_t i = (size_t)y*w + x;
            if (!edgeMask[i]) continue;

            // Local mean over 3x3 of F_E
            float s = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx)
                    s += getPixelF(FE, w, h, x+kx, y+ky);
            float lm = s / 9.0f;

            // Local variance
            float v = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    float d = getPixelF(FE, w, h, x+kx, y+ky) - lm;
                    v += d * d;
                }
            lv[i] = v / 9.0f;
            VRmin = std::min(VRmin, lv[i]);
            VRmax = std::max(VRmax, lv[i]);
        }

    if (VRmin > VRmax) {
        // No edge pixels — F_AES = F_E (pass through)
        FAES = FE;
        return;
    }

    float S = (VRmax - VRmin) / 4.0f;
    if (S < 1e-6f) S = 1.0f;

    // Step 2: For each edge pixel, build H and convolve with F_E
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            size_t i = (size_t)y*w + x;
            if (!edgeMask[i]) {
                FAES[i] = 0.0f;  // non-edge locations: 0 in F_AES
                continue;
            }

            // Select cmp based on variance level (Algorithm 1)
            float lvr = lv[i];
            float cmp;
            if      (lvr > VRmin       && lvr <= VRmin + S)   cmp = 16.0f;
            else if (lvr > VRmin + S   && lvr <= VRmin + 2*S) cmp = 24.0f;
            else if (lvr > VRmin + 2*S && lvr <= VRmin + 3*S) cmp = 32.0f;
            else if (lvr > VRmin + 3*S && lvr <= VRmin + 4*S) cmp = 40.0f;
            else                                                cmp = 48.0f;

            // Build H kernel (sum = 0):
            // center = cmp, all 8 neighbors = -cmp/8
            // Apply convolution: F_AES(m,n) = sum H(k,l)*F_E(m+k, n+l)
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    float hval = (ky == 0 && kx == 0) ? cmp : (-cmp / 8.0f);
                    sum += hval * getPixelF(FE, w, h, x+kx, y+ky);
                }

            FAES[i] = std::max(0.0f, std::min(255.0f, sum));
        }
}

// =============================================================================
// ODAD — Optimized Directional Anisotropic Diffusion (Section 4.3)
//
// Operates on F_s (smooth region, edge pixels = 0/boundary).
// One iteration (t=1) per paper Section 4.3 and Figure 4.
//
// [F_OTP]^{t+1}_{m,n} = [F_s]^t_{m,n} + lambda * sum(d_dir * grad_dir)
//
// d_dir = exp(-(|grad_dir|/K)^2)   [Eq. 11]
// grad_dir = neighbor - center      [Eq. 9]
//
// Edge pixels (edgeMask=true) are treated as fixed boundary conditions:
// their values are read but never updated.
// =============================================================================
static void applyODAD(const std::vector<float>& Fs,
    const std::vector<bool>& edgeMask,
    int w, int h,
    std::vector<float>& FOTP,
    float lambda, float K, int iters)
{
    const float K2 = K * K;
    // Start from F_s (smooth region image)
    FOTP = Fs;

    for (int t = 0; t < iters; ++t) {
        std::vector<float> next = FOTP;
        for (int m = 0; m < h; ++m)
            for (int n = 0; n < w; ++n) {
                size_t i = (size_t)m*w + n;
                // Edge pixels are fixed boundaries — not updated
                if (edgeMask[i]) continue;

                float c = FOTP[i];

                // 8 directional gradients (Eq. 9)
                // Note: edge pixels (value=0 in Fs) act as zero-value boundaries
                // We use the FOTP values which start as Fs
                float gN  = getPixelF(FOTP, w, h, n,   m-1) - c;
                float gS  = getPixelF(FOTP, w, h, n,   m+1) - c;
                float gE  = getPixelF(FOTP, w, h, n+1, m  ) - c;
                float gW  = getPixelF(FOTP, w, h, n-1, m  ) - c;
                float gNE = getPixelF(FOTP, w, h, n+1, m-1) - c;
                float gSE = getPixelF(FOTP, w, h, n+1, m+1) - c;
                float gWS = getPixelF(FOTP, w, h, n-1, m+1) - c;
                float gWN = getPixelF(FOTP, w, h, n-1, m-1) - c;

                // Diffusion coefficients (Eq. 10, 11)
                auto dc = [&](float g) {
                    return std::exp(-(g*g) / K2);
                };

                float upd = lambda * (
                    dc(gN)*gN + dc(gS)*gS + dc(gE)*gE + dc(gW)*gW +
                    dc(gNE)*gNE + dc(gSE)*gSE + dc(gWS)*gWS + dc(gWN)*gWN
                );

                next[i] = std::max(0.0f, std::min(255.0f, c + upd));
            }
        FOTP = next;
    }
}

// =============================================================================
// SSIM helper for CS fitness function
// Computed on smooth region pixels only (where edgeMask = false).
// SSIM(F_OTP, F_s) measures how well ODAD preserves the smooth region's
// structural content. λ=0 gives SSIM=1.0 (no change), larger λ may
// improve or degrade it. We find largest λ with SSIM above threshold.
// =============================================================================
static float computeSSIM(const std::vector<float>& img1,
    const std::vector<float>& img2,
    const std::vector<bool>& edgeMask,
    int w, int h)
{
    // Collect smooth-region pixel pairs
    double mu1 = 0, mu2 = 0;
    int cnt = 0;
    for (size_t i = 0; i < (size_t)w*h; ++i) {
        if (edgeMask[i]) continue;
        mu1 += img1[i];
        mu2 += img2[i];
        ++cnt;
    }
    if (cnt == 0) return 1.0f;
    mu1 /= cnt;
    mu2 /= cnt;

    double sig1sq = 0, sig2sq = 0, sig12 = 0;
    for (size_t i = 0; i < (size_t)w*h; ++i) {
        if (edgeMask[i]) continue;
        double d1 = img1[i] - mu1;
        double d2 = img2[i] - mu2;
        sig1sq += d1 * d1;
        sig2sq += d2 * d2;
        sig12  += d1 * d2;
    }
    sig1sq /= cnt;
    sig2sq /= cnt;
    sig12  /= cnt;

    // SSIM constants (L=255 for [0,255] range)
    const double C1 = (0.01 * 255) * (0.01 * 255);  // (k1*L)^2, k1=0.01
    const double C2 = (0.03 * 255) * (0.03 * 255);  // (k2*L)^2, k2=0.03

    double num = (2*mu1*mu2 + C1) * (2*sig12 + C2);
    double den = (mu1*mu1 + mu2*mu2 + C1) * (sig1sq + sig2sq + C2);

    return (float)(num / den);
}

// =============================================================================
// CUCKOO SEARCH for optimal lambda (Algorithm 2, Section 4.6)
//
// Fitness function: we want the LARGEST lambda that still preserves
// structural similarity of the smooth region (SSIM of F_OTP vs F_s).
//
// Rationale: λ=0 trivially gives SSIM=1 (no change). We want the largest
// λ that diffuses texture details (anisotropically) without degrading SSIM
// below a threshold. This matches the paper's intent: CS finds the
// "stability parameter" (Section 4.6) — the largest stable λ.
//
// Fitness = lambda * SSIM_weight
// where SSIM_weight = 1.0 if SSIM > threshold, else very negative
// This drives the optimizer to find large λ with good SSIM.
// =============================================================================
static float levyStep(std::mt19937& rng, float beta) {
    std::normal_distribution<float> nd(0, 1);
    double num = std::tgamma(1.0 + beta) * std::sin(kPi * beta / 2.0);
    double den = std::tgamma((1.0 + beta) / 2.0) * beta *
                 std::pow(2.0, (beta - 1.0) / 2.0);
    float sig = (float)std::pow(std::abs(num / den), 1.0 / beta);
    float u = nd(rng) * sig;
    float v = nd(rng);
    if (std::abs(v) < 1e-10f) v = 1e-10f;
    return u / std::pow(std::abs(v), 1.0f / beta);
}

static float findLambda(const std::vector<float>& Fs,
    const std::vector<bool>& edgeMask,
    int w, int h,
    const OdasParams& params)
{
    const float lMin = 0.0f;
    const float lMax = kPi / 4.0f;   // π/4 ≈ 0.7854

    // SSIM threshold: we want λ large AND SSIM high
    // Threshold of 0.998 means we allow tiny degradation for larger λ
    const float ssimThreshold = 0.998f;

    const int   n     = params.csNests;
    const int   MAXit = params.csMaxIter;
    const float pa    = params.csPa;
    const float beta  = params.csLevyBeta;

    // Fitness: maximize lambda weighted by SSIM quality
    // If SSIM drops below threshold, heavy penalty.
    auto fitness = [&](float lambda) -> float {
        if (lambda < 1e-6f) return 0.0f;  // λ=0 gives no improvement, score=0

        std::vector<float> FOTP;
        applyODAD(Fs, edgeMask, w, h, FOTP, lambda, params.K,
                  params.odadIterations);

        float ssim = computeSSIM(Fs, FOTP, edgeMask, w, h);

        if (ssim < ssimThreshold) {
            // Penalty: below threshold, heavily penalize
            return -1.0f + ssim;  // negative score
        }
        // Reward larger lambda that still maintains SSIM
        return lambda * ssim;
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ur(lMin, lMax);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // Initialize population
    std::vector<float> nests(n), fits(n);
    for (int i = 0; i < n; ++i) {
        nests[i] = ur(rng);
        fits[i]  = fitness(nests[i]);
    }

    // Best solution
    int bi = (int)(std::max_element(fits.begin(), fits.end()) - fits.begin());
    float bL = nests[bi], bF = fits[bi];

    std::cout << "  [ODAS-CS] n=" << n << " MAXit=" << MAXit
              << " λ_range=[0, π/4=" << lMax << "]" << std::endl;

    for (int t = 0; t < MAXit; ++t) {
        for (int i = 0; i < n; ++i) {
            // Lévy flight: new solution
            float step = levyStep(rng, beta);
            // Step scale: 1% of range, modulated by Lévy step
            float newL = nests[i] + 0.01f * (lMax - lMin) * step;
            newL = std::max(lMin, std::min(lMax, newL));

            float newF = fitness(newL);

            // Choose random nest j to compare
            int j = (int)(u01(rng) * n) % n;
            if (newF > fits[j]) {
                nests[j] = newL;
                fits[j]  = newF;
                if (newF > bF) { bF = newF; bL = newL; }
            }
        }

        // Abandon worst pa fraction of nests, replace with random
        int na = std::max(1, (int)(pa * n));
        std::vector<int> si(n);
        std::iota(si.begin(), si.end(), 0);
        std::sort(si.begin(), si.end(),
                  [&](int a, int b) { return fits[a] < fits[b]; });
        for (int k = 0; k < na; ++k) {
            int idx = si[k];
            nests[idx] = ur(rng);
            fits[idx]  = fitness(nests[idx]);
            if (fits[idx] > bF) { bF = fits[idx]; bL = nests[idx]; }
        }

        if ((t+1) % 10 == 0 || t == MAXit-1)
            std::cout << "  [ODAS-CS] Iter " << (t+1) << "/" << MAXit
                      << "  λ=" << std::fixed << std::setprecision(4) << bL
                      << "  fit=" << bF << std::endl;
    }

    // If no λ > 0 was found with good SSIM, use a safe small value
    if (bL < 1e-6f) {
        bL = lMax * 0.1f;  // fallback: 10% of max
        std::cout << "  [ODAS-CS] Fallback λ=" << bL << std::endl;
    }

    std::cout << "  [ODAS-CS] Optimal λ=" << std::fixed
              << std::setprecision(4) << bL << std::endl;
    return bL;
}

// =============================================================================
// IHR COMPOSITION (Eq. 12)
// F_IHR = F_AES + F_OTP
//
// F_AES: non-zero at edge locations (sharpened edge values)
// F_OTP: non-zero at smooth locations (ODAD-filtered smooth values)
// Their sum reconstructs the full image with enhancements.
// =============================================================================
static void composeIHR(const std::vector<float>& FAES,
    const std::vector<float>& FOTP,
    int w, int h,
    std::vector<float>& FIHR)
{
    size_t N = (size_t)w * h;
    FIHR.resize(N);
    for (size_t i = 0; i < N; ++i)
        FIHR[i] = std::max(0.0f, std::min(255.0f, FAES[i] + FOTP[i]));
}

// =============================================================================
// RESIDUAL SHARPENING (Eqs. 13, 14)
// F_BHR = Lanczos3_Downscale(Lanczos3_Upscale(F_IHR, s), s)
// F_RES = F_IHR - F_BHR
// F_RHR = F_IHR + F_RES = 2*F_IHR - F_BHR
// =============================================================================
static void residual(const std::vector<float>& FIHR,
    int w, int h, int sf,
    std::vector<float>& FRHR)
{
    int uw = w * sf, uh = h * sf;
    std::vector<float> up, bhr;
    L3up(FIHR, w, h, up, uw, uh);
    L3dn(up, uw, uh, bhr, w, h);

    size_t N = (size_t)w * h;
    FRHR.resize(N);
    for (size_t i = 0; i < N; ++i)
        FRHR[i] = std::max(0.0f, std::min(255.0f,
            FIHR[i] + (FIHR[i] - bhr[i])));  // = 2*FIHR - BHR
}

// =============================================================================
// COLOR CONVERSION HELPERS
// =============================================================================
static void rgb2ycbcr(const unsigned char* rgb, int w, int h,
    std::vector<float>& Y, std::vector<float>& Cb, std::vector<float>& Cr)
{
    size_t N = (size_t)w * h;
    Y.resize(N); Cb.resize(N); Cr.resize(N);
    for (size_t i = 0; i < N; ++i) {
        float r = rgb[i*4+0], g = rgb[i*4+1], b = rgb[i*4+2];
        Y[i]  =  0.299f*r + 0.587f*g + 0.114f*b;
        Cb[i] = -0.16874f*r - 0.33126f*g + 0.5f*b + 128.0f;
        Cr[i] =  0.5f*r - 0.41869f*g - 0.08131f*b + 128.0f;
    }
}

static void bilinUp(const std::vector<float>& src, int iw, int ih,
    std::vector<float>& dst, int ow, int oh)
{
    dst.resize((size_t)ow * oh);
    float sx = (float)iw / ow, sy = (float)ih / oh;
    for (int oy = 0; oy < oh; ++oy) {
        float fy = ((float)oy + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        float v = fy - y0;
        for (int ox = 0; ox < ow; ++ox) {
            float fx = ((float)ox + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(fx);
            float u = fx - x0;
            float p00 = getPixelF(src, iw, ih, x0,   y0);
            float p10 = getPixelF(src, iw, ih, x0+1, y0);
            float p01 = getPixelF(src, iw, ih, x0,   y0+1);
            float p11 = getPixelF(src, iw, ih, x0+1, y0+1);
            dst[(size_t)oy*ow + ox] =
                (p00 + u*(p10-p00)) * (1-v) + (p01 + u*(p11-p01)) * v;
        }
    }
}

static void bilinUpRGBA(const unsigned char* src, int iw, int ih,
    unsigned char* dst, int ow, int oh)
{
    float sx = (float)iw / ow, sy = (float)ih / oh;
    for (int oy = 0; oy < oh; ++oy) {
        float fy = ((float)oy + 0.5f) * sy - 0.5f;
        int y0 = std::max(0, std::min(ih-1, (int)std::floor(fy)));
        int y1 = std::max(0, std::min(ih-1, y0+1));
        float v = fy - std::floor(fy);
        for (int ox = 0; ox < ow; ++ox) {
            float fx = ((float)ox + 0.5f) * sx - 0.5f;
            int x0 = std::max(0, std::min(iw-1, (int)std::floor(fx)));
            int x1 = std::max(0, std::min(iw-1, x0+1));
            float u = fx - std::floor(fx);
            for (int c = 0; c < 4; ++c) {
                float p00 = src[(y0*iw+x0)*4+c], p10 = src[(y0*iw+x1)*4+c];
                float p01 = src[(y1*iw+x0)*4+c], p11 = src[(y1*iw+x1)*4+c];
                dst[(oy*ow+ox)*4+c] = clampByte(
                    (p00 + u*(p10-p00)) * (1-v) + (p01 + u*(p11-p01)) * v);
            }
        }
    }
}

static void ycbcr2rgba(const std::vector<float>& Y,
    const std::vector<float>& Cb, const std::vector<float>& Cr,
    const unsigned char* al, unsigned char* dst, int w, int h)
{
    for (size_t i = 0; i < (size_t)w*h; ++i) {
        float y = Y[i], cb = Cb[i]-128.0f, cr = Cr[i]-128.0f;
        dst[i*4+0] = clampByte(y + 1.402f*cr);
        dst[i*4+1] = clampByte(y - 0.34414f*cb - 0.71414f*cr);
        dst[i*4+2] = clampByte(y + 1.772f*cb);
        dst[i*4+3] = al[i*4+3];
    }
}

// =============================================================================
// MAIN PIPELINE (per channel)
// =============================================================================
static std::vector<float> odasPipeline(
    const std::vector<float>& ch,
    int inW, int inH, int outW, int outH,
    int sf,
    const OdasParams& params,
    bool verbose)
{
    // ── Step 1: Lanczos3 Upscale ──────────────────────────────────────────
    if (verbose) std::cout << "  [ODAS] Step 1: Lanczos3 upscale..." << std::endl;
    std::vector<float> Fhat;
    L3up(ch, inW, inH, Fhat, outW, outH);

    // ── Step 2: Edge Detection + F_E / F_s split ─────────────────────────
    if (verbose) std::cout << "  [ODAS] Step 2: Canny edge detection..." << std::endl;
    std::vector<bool>  edgeMask;
    std::vector<float> FE, Fs;
    cannyAndSplit(Fhat, outW, outH, edgeMask, FE, Fs,
                  params.cannyLowThresh, params.cannyHighThresh);
    if (verbose) {
        size_t ec = std::count(edgeMask.begin(), edgeMask.end(), true);
        std::cout << "  [ODAS] Edge pixels: " << ec << "/" << (size_t)outW*outH
                  << " (" << std::fixed << std::setprecision(2)
                  << 100.0f * ec / (outW*outH) << "%)" << std::endl;
    }

    // ── Step 3: AES on F_E ───────────────────────────────────────────────
    if (verbose) std::cout << "  [ODAS] Step 3: AES (convolve H with F_E)..." << std::endl;
    std::vector<float> FAES;
    applyAES(FE, edgeMask, outW, outH, FAES);
    if (verbose) {
        double chg = 0; int cnt = 0;
        for (size_t i = 0; i < (size_t)outW*outH; ++i)
            if (edgeMask[i]) { chg += std::abs(FAES[i] - FE[i]); ++cnt; }
        std::cout << "  [ODAS] AES mean edge change: "
                  << (cnt > 0 ? chg/cnt : 0.0) << " grey levels" << std::endl;
    }

    // ── Step 4: Cuckoo Search for λ ──────────────────────────────────────
    if (verbose) std::cout << "  [ODAS] Step 4: Cuckoo Search for λ..." << std::endl;
    float lambda = findLambda(Fs, edgeMask, outW, outH, params);

    // ── Step 5: ODAD on F_s ──────────────────────────────────────────────
    if (verbose)
        std::cout << "  [ODAS] Step 5: ODAD filter (λ="
                  << std::fixed << std::setprecision(4) << lambda
                  << ", K=" << params.K << ")..." << std::endl;
    std::vector<float> FOTP;
    applyODAD(Fs, edgeMask, outW, outH, FOTP, lambda, params.K,
              params.odadIterations);
    if (verbose) {
        double chg = 0; int cnt = 0;
        for (size_t i = 0; i < (size_t)outW*outH; ++i)
            if (!edgeMask[i]) { chg += std::abs(FOTP[i] - Fs[i]); ++cnt; }
        std::cout << "  [ODAS] ODAD mean smooth change: "
                  << (cnt > 0 ? chg/cnt : 0.0) << " grey levels" << std::endl;
    }

    // ── Step 6: IHR = F_AES + F_OTP ─────────────────────────────────────
    if (verbose) std::cout << "  [ODAS] Step 6: IHR = F_AES + F_OTP..." << std::endl;
    std::vector<float> FIHR;
    composeIHR(FAES, FOTP, outW, outH, FIHR);

    // ── Step 7: Residual sharpening ──────────────────────────────────────
    if (verbose) std::cout << "  [ODAS] Step 7: Residual sharpening..." << std::endl;
    std::vector<float> FRHR;
    residual(FIHR, outW, outH, sf, FRHR);
    if (verbose) {
        double chg = 0;
        for (size_t i = 0; i < (size_t)outW*outH; ++i)
            chg += std::abs(FRHR[i] - FIHR[i]);
        std::cout << "  [ODAS] Residual mean change: "
                  << chg / (outW*outH) << " grey levels" << std::endl;
    }

    return FRHR;
}

// =============================================================================
// PUBLIC ENTRY POINT
// =============================================================================
void scaleODAS(const unsigned char* input, int inW, int inH,
    unsigned char* output, int outW, int outH,
    const OdasParams& params)
{
    int sf = std::max(1, (int)std::round((float)outW / inW));
    std::cout << "  [ODAS] Input:  " << inW << "x" << inH << std::endl;
    std::cout << "  [ODAS] Output: " << outW << "x" << outH << std::endl;
    std::cout << "  [ODAS] Scale:  ~" << sf << "x  K=" << params.K
              << "  iters=" << params.odadIterations << std::endl;

    // Bilinear upscale for alpha channel reference
    std::vector<unsigned char> bilin((size_t)outW * outH * 4);
    bilinUpRGBA(input, inW, inH, bilin.data(), outW, outH);

    if (params.useYCbCr) {
        // Process Y channel through full pipeline; Cb/Cr upscaled bilinearly
        std::vector<float> Y, Cb, Cr;
        rgb2ycbcr(input, inW, inH, Y, Cb, Cr);

        auto FRHR = odasPipeline(Y, inW, inH, outW, outH, sf, params, true);

        std::vector<float> CbO, CrO;
        bilinUp(Cb, inW, inH, CbO, outW, outH);
        bilinUp(Cr, inW, inH, CrO, outW, outH);

        std::cout << "  [ODAS] Step 8: RGBA output..." << std::endl;
        ycbcr2rgba(FRHR, CbO, CrO, bilin.data(), output, outW, outH);
    } else {
        std::cout << "  [ODAS] RGB mode" << std::endl;
        std::vector<std::vector<float>> ch(3);
        for (int c = 0; c < 3; ++c) {
            std::vector<float> ci((size_t)inW * inH);
            for (int i = 0; i < inW*inH; ++i) ci[i] = input[i*4+c];
            ch[c] = odasPipeline(ci, inW, inH, outW, outH, sf, params, c == 0);
        }
        for (int i = 0; i < outW*outH; ++i) {
            output[i*4+0] = clampByte(ch[0][i]);
            output[i*4+1] = clampByte(ch[1][i]);
            output[i*4+2] = clampByte(ch[2][i]);
            output[i*4+3] = bilin[i*4+3];
        }
    }
    std::cout << "  [ODAS] Done." << std::endl;
}
