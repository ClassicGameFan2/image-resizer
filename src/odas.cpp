// =============================================================================
// odas.cpp — v8: Correct boundary conditions for AES and ODAD
//
// KEY FIXES vs v7:
//   1. AES: convolve H with F_hat neighborhood (not zeroed F_E sparse image).
//      F_E pixel values at edge locations, reading full F_hat for neighbors.
//      This prevents the 182 grey level explosion.
//
//   2. ODAD: smooth pixels read F_hat at edge neighbor locations (fixed
//      boundary condition), not zero. This prevents artificial gradients
//      that darken/blur regions adjacent to edges.
//
//   3. F_E and F_s are now only used as MASKS to know which pixels are
//      edge vs smooth. The actual pixel VALUES always come from F_hat
//      (or FOTP for smooth-to-smooth neighbors in ODAD).
//
// PIPELINE (matches paper Section 4 exactly):
//   F_hat  = Lanczos3_Upscale(F_LR)
//   edgeMask = Canny(F_hat)          -- boolean mask
//   F_AES[edge]  = Convolve(H, F_hat) at edge locations   [Eq. 5, Alg 1]
//   F_AES[smooth]= 0
//   F_OTP[smooth]= ODAD(F_hat_smooth, boundary=F_hat_edge) [Eq. 8-11]
//   F_OTP[edge]  = 0
//   F_IHR = F_AES + F_OTP            [Eq. 12]
//   F_BHR = L3dn(L3up(F_IHR))
//   F_RHR = 2*F_IHR - F_BHR         [Eqs. 13-14]
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
// LANCZOS3
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
// CANNY — returns boolean edge mask only
// F_hat values are used directly; we just need to know WHICH pixels are edges.
// =============================================================================
static void gblur5(const std::vector<float>& src, int w, int h,
    std::vector<float>& dst)
{
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

static void cannyEdgeMask(const std::vector<float>& Fhat,
    int w, int h,
    std::vector<bool>& edgeMask,
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

    edgeMask.assign(N, false);
    for (int i = 0; i < N; ++i)
        edgeMask[i] = (st[i] == 2);
}

// =============================================================================
// AES — Adaptive Edge Sharpening (Algorithm 1, Section 4.2)
//
// For each edge pixel (m,n):
//   1. Compute local variance in 3x3 neighborhood of F_hat (not zeroed F_E)
//   2. Select cmp from {16,24,32,40,48} based on variance level
//   3. Build H kernel: center=cmp, neighbors=-cmp/8
//   4. Convolve H with F_hat neighborhood → F_AES(m,n)
//
// CRITICAL: Convolution reads from F_hat (full image), NOT from a sparse
// zeroed mask. This prevents the 182 grey level explosion.
//
// F_AES is non-zero only at edge pixel locations.
// =============================================================================
static void applyAES(const std::vector<float>& Fhat,
    const std::vector<bool>& edgeMask,
    int w, int h,
    std::vector<float>& FAES)
{
    size_t N = (size_t)w * h;
    FAES.assign(N, 0.0f);

    // Step 1: Compute local variance for all edge pixels using F_hat
    std::vector<float> lv(N, 0.0f);
    float VRmin = 1e30f, VRmax = -1e30f;

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            size_t i = (size_t)y*w + x;
            if (!edgeMask[i]) continue;

            // Local mean from F_hat
            float s = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx)
                    s += getPixelF(Fhat, w, h, x+kx, y+ky);
            float lm = s / 9.0f;

            // Local variance from F_hat
            float v = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    float d = getPixelF(Fhat, w, h, x+kx, y+ky) - lm;
                    v += d * d;
                }
            lv[i] = v / 9.0f;
            VRmin = std::min(VRmin, lv[i]);
            VRmax = std::max(VRmax, lv[i]);
        }

    if (VRmin > VRmax) {
        // No edge pixels
        FAES.assign(N, 0.0f);
        return;
    }

    float S = (VRmax - VRmin) / 4.0f;
    if (S < 1e-6f) S = 1.0f;

    // Step 2: Convolve H with F_hat at each edge pixel location
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            size_t i = (size_t)y*w + x;
            if (!edgeMask[i]) {
                FAES[i] = 0.0f;  // non-edge: zero contribution
                continue;
            }

            // Select cmp from Algorithm 1
            float lvr = lv[i];
            float cmp;
            if      (lvr > VRmin       && lvr <= VRmin +   S) cmp = 16.0f;
            else if (lvr > VRmin +   S && lvr <= VRmin + 2*S) cmp = 24.0f;
            else if (lvr > VRmin + 2*S && lvr <= VRmin + 3*S) cmp = 32.0f;
            else if (lvr > VRmin + 3*S && lvr <= VRmin + 4*S) cmp = 40.0f;
            else                                                cmp = 48.0f;

            // Convolve H (center=cmp, neighbors=-cmp/8) with F_hat neighborhood
            // H sums to zero: cmp + 8*(-cmp/8) = 0
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    float hval = (ky == 0 && kx == 0) ? cmp : (-cmp / 8.0f);
                    sum += hval * getPixelF(Fhat, w, h, x+kx, y+ky);
                }

            // sum is the Laplacian response — this IS the sharpened edge value
            // The paper's F_AES is the convolution output, which represents
            // the high-frequency edge detail extracted from F_hat.
            // We need to ADD it back to the edge pixel to get the sharpened value:
            // F_AES_final = F_hat[edge] + sum  (unsharp mask interpretation)
            // BUT: the paper says F_IHR = F_AES + F_OTP and F_OTP covers smooth
            // pixels (=0 at edge locations). So F_AES must contain the full
            // sharpened pixel value at edge locations, not just the delta.
            //
            // Correct interpretation: F_AES = F_hat + Laplacian_response
            // where Laplacian_response = convolution of H with F_hat
            // H has center=cmp and neighbors=-cmp/8, so:
            // response = cmp*(center - avg_neighbors) = cmp * local_laplacian
            // F_AES = F_hat[edge] + response   (sharpened edge pixel)
            FAES[i] = std::max(0.0f, std::min(255.0f,
                Fhat[i] + sum));
        }
}

// =============================================================================
// ODAD — Optimized Directional Anisotropic Diffusion (Section 4.3)
//
// Operates on smooth region pixels. Edge pixels are FIXED BOUNDARIES
// and use F_hat values (not zero) when read by smooth pixel neighbors.
//
// This is the critical fix: when a smooth pixel reads a neighbor that
// is an edge pixel, it reads F_hat[neighbor], not 0.
// This prevents large artificial gradients at edge/smooth boundaries.
//
// We maintain two arrays:
//   - FOTP: current state (initialized to F_hat for ALL pixels)
//   - Only smooth pixels are updated each iteration
//   - Edge pixels stay fixed at F_hat values throughout
// =============================================================================
static void applyODAD(const std::vector<float>& Fhat,
    const std::vector<bool>& edgeMask,
    int w, int h,
    std::vector<float>& FOTP,
    float lambda, float K, int iters)
{
    const float K2 = K * K;

    // Initialize FOTP to F_hat for ALL pixels (both edge and smooth).
    // Edge pixels remain at F_hat throughout (fixed boundary).
    // Smooth pixels will be updated by diffusion.
    FOTP = Fhat;

    for (int t = 0; t < iters; ++t) {
        std::vector<float> next = FOTP;
        for (int m = 0; m < h; ++m)
            for (int n = 0; n < w; ++n) {
                size_t i = (size_t)m*w + n;
                // Edge pixels: fixed boundary, not updated
                if (edgeMask[i]) continue;

                float c = FOTP[i];

                // Read neighbors from FOTP (which has F_hat at edge pixels,
                // updated smooth values at smooth pixels). This ensures
                // edge pixels act as proper fixed boundaries.
                float gN  = getPixelF(FOTP, w, h, n,   m-1) - c;
                float gS  = getPixelF(FOTP, w, h, n,   m+1) - c;
                float gE  = getPixelF(FOTP, w, h, n+1, m  ) - c;
                float gW  = getPixelF(FOTP, w, h, n-1, m  ) - c;
                float gNE = getPixelF(FOTP, w, h, n+1, m-1) - c;
                float gSE = getPixelF(FOTP, w, h, n+1, m+1) - c;
                float gWS = getPixelF(FOTP, w, h, n-1, m+1) - c;
                float gWN = getPixelF(FOTP, w, h, n-1, m-1) - c;

                // Diffusion coefficients (Eq. 11)
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
    // After ODAD: edge pixels in FOTP = F_hat values (unchanged)
    // For IHR composition we need F_OTP to have 0 at edge locations
    // (since F_AES covers edges, F_OTP covers smooth).
    // Zero out edge pixels in FOTP so F_IHR = F_AES + F_OTP works correctly.
    for (size_t i = 0; i < (size_t)w*h; ++i)
        if (edgeMask[i]) FOTP[i] = 0.0f;
}

// =============================================================================
// SSIM for CS fitness (smooth region only)
// =============================================================================
static float computeSSIM(const std::vector<float>& img1,
    const std::vector<float>& img2,
    const std::vector<bool>& edgeMask,
    int w, int h)
{
    double mu1 = 0, mu2 = 0;
    int cnt = 0;
    for (size_t i = 0; i < (size_t)w*h; ++i) {
        if (edgeMask[i]) continue;
        mu1 += img1[i];
        mu2 += img2[i];
        ++cnt;
    }
    if (cnt == 0) return 1.0f;
    mu1 /= cnt; mu2 /= cnt;

    double sig1sq = 0, sig2sq = 0, sig12 = 0;
    for (size_t i = 0; i < (size_t)w*h; ++i) {
        if (edgeMask[i]) continue;
        double d1 = img1[i] - mu1;
        double d2 = img2[i] - mu2;
        sig1sq += d1*d1;
        sig2sq += d2*d2;
        sig12  += d1*d2;
    }
    sig1sq /= cnt; sig2sq /= cnt; sig12 /= cnt;

    const double C1 = (0.01*255)*(0.01*255);
    const double C2 = (0.03*255)*(0.03*255);
    double num = (2*mu1*mu2 + C1) * (2*sig12 + C2);
    double den = (mu1*mu1 + mu2*mu2 + C1) * (sig1sq + sig2sq + C2);
    return (float)(num / den);
}

// =============================================================================
// CUCKOO SEARCH for optimal lambda (Algorithm 2, Section 4.6)
//
// Now that ODAD uses F_hat as boundary (not zero), the fitness function
// correctly measures smooth-region structural preservation.
// =============================================================================
static float levyStep(std::mt19937& rng, float beta) {
    std::normal_distribution<float> nd(0, 1);
    double num = std::tgamma(1.0+beta) * std::sin(kPi*beta/2.0);
    double den = std::tgamma((1.0+beta)/2.0) * beta *
                 std::pow(2.0, (beta-1.0)/2.0);
    float sig = (float)std::pow(std::abs(num/den), 1.0/beta);
    float u = nd(rng) * sig;
    float v = nd(rng);
    if (std::abs(v) < 1e-10f) v = 1e-10f;
    return u / std::pow(std::abs(v), 1.0f/beta);
}

static float findLambda(const std::vector<float>& Fhat,
    const std::vector<bool>& edgeMask,
    int w, int h,
    const OdasParams& params)
{
    const float lMin = 0.0f;
    const float lMax = kPi / 4.0f;
    const float ssimThreshold = 0.998f;

    const int   n     = params.csNests;
    const int   MAXit = params.csMaxIter;
    const float pa    = params.csPa;
    const float beta  = params.csLevyBeta;

    // Fitness: maximize lambda while keeping SSIM(ODAD_output, Fhat_smooth) high
    // λ=0 gives score=0 (no diffusion, no benefit)
    // Large λ with good SSIM gives high score
    auto fitness = [&](float lambda) -> float {
        if (lambda < 1e-6f) return 0.0f;

        std::vector<float> FOTP;
        applyODAD(Fhat, edgeMask, w, h, FOTP, lambda,
                  params.K, params.odadIterations);

        // Restore smooth values for SSIM (FOTP has 0 at edges after applyODAD)
        // Compare FOTP smooth values vs Fhat smooth values
        float ssim = computeSSIM(Fhat, FOTP, edgeMask, w, h);

        if (ssim < ssimThreshold)
            return -1.0f + ssim;  // penalty

        return lambda * ssim;     // maximize large lambda with good SSIM
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ur(lMin, lMax);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    std::vector<float> nests(n), fits(n);
    for (int i = 0; i < n; ++i) {
        nests[i] = ur(rng);
        fits[i]  = fitness(nests[i]);
    }

    int bi = (int)(std::max_element(fits.begin(), fits.end()) - fits.begin());
    float bL = nests[bi], bF = fits[bi];

    std::cout << "  [ODAS-CS] n=" << n << " MAXit=" << MAXit
              << " λ_range=[0, π/4=" << std::fixed << std::setprecision(4)
              << lMax << "]" << std::endl;

    for (int t = 0; t < MAXit; ++t) {
        for (int i = 0; i < n; ++i) {
            float step = levyStep(rng, beta);
            float newL = nests[i] + 0.01f * (lMax - lMin) * step;
            newL = std::max(lMin, std::min(lMax, newL));
            float newF = fitness(newL);
            int j = (int)(u01(rng) * n) % n;
            if (newF > fits[j]) {
                nests[j] = newL;
                fits[j]  = newF;
                if (newF > bF) { bF = newF; bL = newL; }
            }
        }

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

    if (bL < 1e-6f) {
        bL = lMax * 0.5f;
        std::cout << "  [ODAS-CS] Fallback λ=" << bL << std::endl;
    }

    std::cout << "  [ODAS-CS] Optimal λ=" << std::fixed
              << std::setprecision(4) << bL << std::endl;
    return bL;
}

// =============================================================================
// IHR COMPOSITION (Eq. 12): F_IHR = F_AES + F_OTP
// F_AES: sharpened edge values at edge locations, 0 at smooth
// F_OTP: diffused smooth values at smooth locations, 0 at edges
// Sum gives full image with both regions enhanced.
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
// RESIDUAL SHARPENING (Eqs. 13-14)
// F_RHR = 2*F_IHR - F_BHR
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
            FIHR[i] + (FIHR[i] - bhr[i])));
}

// =============================================================================
// COLOR HELPERS
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
                (p00 + u*(p10-p00))*(1-v) + (p01 + u*(p11-p01))*v;
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
                    (p00+u*(p10-p00))*(1-v) + (p01+u*(p11-p01))*v);
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
// PIPELINE (per channel)
// =============================================================================
static std::vector<float> odasPipeline(
    const std::vector<float>& ch,
    int inW, int inH, int outW, int outH, int sf,
    const OdasParams& params, bool verbose)
{
    // Step 1: Lanczos3 upscale → F_hat
    if (verbose) std::cout << "  [ODAS] Step 1: Lanczos3 upscale..." << std::endl;
    std::vector<float> Fhat;
    L3up(ch, inW, inH, Fhat, outW, outH);

    // Step 2: Canny edge detection → boolean mask only
    if (verbose) std::cout << "  [ODAS] Step 2: Canny edge detection..." << std::endl;
    std::vector<bool> edgeMask;
    cannyEdgeMask(Fhat, outW, outH, edgeMask,
                  params.cannyLowThresh, params.cannyHighThresh);
    if (verbose) {
        size_t ec = std::count(edgeMask.begin(), edgeMask.end(), true);
        std::cout << "  [ODAS] Edge pixels: " << ec << "/" << (size_t)outW*outH
                  << " (" << std::fixed << std::setprecision(2)
                  << 100.0f*ec/(outW*outH) << "%)" << std::endl;
    }

    // Step 3: AES — convolve H with F_hat at edge locations
    if (verbose) std::cout << "  [ODAS] Step 3: AES (F_hat neighborhood)..." << std::endl;
    std::vector<float> FAES;
    applyAES(Fhat, edgeMask, outW, outH, FAES);
    if (verbose) {
        double chg = 0; int cnt = 0;
        for (size_t i = 0; i < (size_t)outW*outH; ++i)
            if (edgeMask[i]) { chg += std::abs(FAES[i] - Fhat[i]); ++cnt; }
        std::cout << "  [ODAS] AES mean edge change: "
                  << (cnt > 0 ? chg/cnt : 0.0) << " grey levels" << std::endl;
    }

    // Step 4: Cuckoo Search for optimal lambda
    if (verbose) std::cout << "  [ODAS] Step 4: Cuckoo Search for λ..." << std::endl;
    float lambda = findLambda(Fhat, edgeMask, outW, outH, params);

    // Step 5: ODAD — F_hat initialized everywhere, smooth pixels updated
    if (verbose)
        std::cout << "  [ODAS] Step 5: ODAD (λ="
                  << std::fixed << std::setprecision(4) << lambda
                  << ", K=" << params.K << ")..." << std::endl;
    std::vector<float> FOTP;
    applyODAD(Fhat, edgeMask, outW, outH, FOTP, lambda,
              params.K, params.odadIterations);
    if (verbose) {
        double chg = 0; int cnt = 0;
        for (size_t i = 0; i < (size_t)outW*outH; ++i)
            if (!edgeMask[i]) { chg += std::abs(FOTP[i] - Fhat[i]); ++cnt; }
        std::cout << "  [ODAS] ODAD mean smooth change: "
                  << (cnt > 0 ? chg/cnt : 0.0) << " grey levels" << std::endl;
    }

    // Step 6: F_IHR = F_AES + F_OTP
    if (verbose) std::cout << "  [ODAS] Step 6: IHR = F_AES + F_OTP..." << std::endl;
    std::vector<float> FIHR;
    composeIHR(FAES, FOTP, outW, outH, FIHR);

    // Step 7: Residual sharpening
    if (verbose) std::cout << "  [ODAS] Step 7: Residual sharpening..." << std::endl;
    std::vector<float> FRHR;
    residual(FIHR, outW, outH, sf, FRHR);
    if (verbose) {
        double chg = 0;
        for (size_t i = 0; i < (size_t)outW*outH; ++i)
            chg += std::abs(FRHR[i] - FIHR[i]);
        std::cout << "  [ODAS] Residual mean change: "
                  << chg/(outW*outH) << " grey levels" << std::endl;
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

    std::vector<unsigned char> bilin((size_t)outW*outH*4);
    bilinUpRGBA(input, inW, inH, bilin.data(), outW, outH);

    if (params.useYCbCr) {
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
            std::vector<float> ci((size_t)inW*inH);
            for (int i = 0; i < inW*inH; ++i) ci[i] = input[i*4+c];
            ch[c] = odasPipeline(ci, inW, inH, outW, outH, sf, params, c==0);
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
