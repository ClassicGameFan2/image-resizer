// =============================================================================
// odas.cpp — v9: Correct unsharp masking framework
//
// FRAMEWORK UNDERSTANDING:
//   F_AES = H convolved with F_hat at edge locations (Laplacian HF response)
//           This is the HIGH-FREQUENCY DETAIL extracted from edges.
//           Values are small (positive/negative Laplacian response).
//
//   F_OTP = ODAD applied to F_hat smooth region.
//           Edge pixels stay fixed at F_hat values (boundary conditions).
//           F_OTP contains F_hat values at ALL locations.
//
//   F_IHR = F_AES + F_OTP
//           = (HF edge detail) + (ODAD-smoothed full image)
//           = unsharp-masked result: smoothed base + sharpened edges
//
// KEY FIXES vs v8:
//   1. Do NOT zero edge pixels in F_OTP. F_OTP = F_hat everywhere,
//      with smooth pixels ODAD-filtered. Edge pixels stay at F_hat.
//   2. F_AES contains Laplacian HF only (can be negative). Adding it
//      to F_OTP (which has F_hat at edges) gives sharpened edge pixels.
//   3. F_AES is zero at smooth pixel locations.
//   4. This matches the unsharp masking equation:
//      sharpened = original + alpha * laplacian
//      where F_OTP provides "original" and F_AES provides "alpha*laplacian"
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
// CANNY — boolean edge mask only
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
    int w, int h, std::vector<bool>& edgeMask, float lo, float hi)
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
            if      (a < 22.5f  || a >= 157.5f) { m1=gm[i+1];           m2=gm[i-1]; }
            else if (a < 67.5f)                  { m1=gm[i-(size_t)w+1]; m2=gm[i+(size_t)w-1]; }
            else if (a < 112.5f)                 { m1=gm[i-(size_t)w];   m2=gm[i+(size_t)w]; }
            else                                 { m1=gm[i-(size_t)w-1]; m2=gm[i+(size_t)w+1]; }
            sup[i] = (m >= m1 && m >= m2) ? m : 0;
        }

    std::vector<int> st(N, 0);
    for (int i = 0; i < N; ++i) {
        if      (sup[i] >= hi) st[i] = 2;
        else if (sup[i] >= lo) st[i] = 1;
    }
    std::vector<int> stk;
    stk.reserve(N/8);
    for (int i = 0; i < N; ++i)
        if (st[i] == 2) stk.push_back(i);
    while (!stk.empty()) {
        int idx = stk.back(); stk.pop_back();
        int px = idx%w, py = idx/w;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (!dx && !dy) continue;
                int nx=px+dx, ny=py+dy;
                if (nx<0||nx>=w||ny<0||ny>=h) continue;
                int ni=ny*w+nx;
                if (st[ni]==1) { st[ni]=2; stk.push_back(ni); }
            }
    }
    edgeMask.assign(N, false);
    for (int i = 0; i < N; ++i) edgeMask[i] = (st[i]==2);
}

// =============================================================================
// AES — Adaptive Edge Sharpening (Algorithm 1)
//
// Computes the Laplacian HIGH-FREQUENCY RESPONSE at edge pixel locations.
// This is an ADDITIVE term (can be positive or negative).
//
// H kernel (sum=0): center=cmp, neighbors=-cmp/8
// F_AES(m,n) = sum_{k,l} H(k,l) * Fhat(m+k, n+l)   [at edge locations]
//            = cmp*(Fhat[center] - avg_of_8_neighbors)
//            = cmp * local_laplacian
//
// This is the high-frequency detail that will be ADDED to F_OTP.
// F_AES is zero at smooth pixel locations.
//
// The Laplacian response of a flat region = 0 (correct, no change).
// The Laplacian response at an edge = amplifies the edge contrast.
// =============================================================================
static void applyAES(const std::vector<float>& Fhat,
    const std::vector<bool>& edgeMask,
    int w, int h,
    std::vector<float>& FAES)
{
    size_t N = (size_t)w * h;
    FAES.assign(N, 0.0f);

    // Step 1: Local variance at edge pixels (from Fhat)
    std::vector<float> lv(N, 0.0f);
    float VRmin = 1e30f, VRmax = -1e30f;

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            size_t i = (size_t)y*w + x;
            if (!edgeMask[i]) continue;

            float s = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx)
                    s += getPixelF(Fhat, w, h, x+kx, y+ky);
            float lm = s / 9.0f;

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

    if (VRmin > VRmax) return;  // no edge pixels, FAES stays zero

    float S = (VRmax - VRmin) / 4.0f;
    if (S < 1e-6f) S = 1.0f;

    // Step 2: Compute Laplacian HF response at each edge pixel
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            size_t i = (size_t)y*w + x;
            if (!edgeMask[i]) continue;  // zero at smooth locations

            float lvr = lv[i];
            float cmp;
            if      (lvr > VRmin       && lvr <= VRmin +   S) cmp = 16.0f;
            else if (lvr > VRmin +   S && lvr <= VRmin + 2*S) cmp = 24.0f;
            else if (lvr > VRmin + 2*S && lvr <= VRmin + 3*S) cmp = 32.0f;
            else if (lvr > VRmin + 3*S && lvr <= VRmin + 4*S) cmp = 40.0f;
            else                                                cmp = 48.0f;

            // H*Fhat = cmp*center - (cmp/8)*sum_of_8_neighbors
            //        = cmp * (center - avg_neighbors)   [the Laplacian]
            // This is the HF response: positive at bright edges, negative at dark
            float center = Fhat[i];
            float neighborSum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    if (ky == 0 && kx == 0) continue;
                    neighborSum += getPixelF(Fhat, w, h, x+kx, y+ky);
                }
            // Laplacian response (the HF component)
            float laplacian = center - neighborSum / 8.0f;
            FAES[i] = cmp * laplacian;
            // Note: NOT clamped here — it's an additive HF term,
            // clamping happens after F_IHR = F_AES + F_OTP
        }
}

// =============================================================================
// ODAD — Optimized Directional Anisotropic Diffusion (Section 4.3)
//
// Initializes FOTP = Fhat for ALL pixels.
// Updates ONLY smooth pixels via anisotropic diffusion.
// Edge pixels remain at Fhat values (fixed boundary conditions).
//
// After this function:
//   FOTP[edge]  = Fhat[edge]   (unchanged — boundary)
//   FOTP[smooth]= diffused value (slightly smoothed anisotropically)
//
// Do NOT zero edge pixels — they are needed for F_IHR = F_AES + F_OTP.
// The F_AES term provides the sharpening delta at edge locations.
// =============================================================================
static void applyODAD(const std::vector<float>& Fhat,
    const std::vector<bool>& edgeMask,
    int w, int h,
    std::vector<float>& FOTP,
    float lambda, float K, int iters)
{
    const float K2 = K * K;
    // Initialize to Fhat for ALL pixels (edge and smooth)
    FOTP = Fhat;

    for (int t = 0; t < iters; ++t) {
        std::vector<float> next = FOTP;
        for (int m = 0; m < h; ++m)
            for (int n = 0; n < w; ++n) {
                size_t i = (size_t)m*w + n;
                if (edgeMask[i]) continue;  // edge = fixed boundary

                float c = FOTP[i];

                float gN  = getPixelF(FOTP, w, h, n,   m-1) - c;
                float gS  = getPixelF(FOTP, w, h, n,   m+1) - c;
                float gE  = getPixelF(FOTP, w, h, n+1, m  ) - c;
                float gW  = getPixelF(FOTP, w, h, n-1, m  ) - c;
                float gNE = getPixelF(FOTP, w, h, n+1, m-1) - c;
                float gSE = getPixelF(FOTP, w, h, n+1, m+1) - c;
                float gWS = getPixelF(FOTP, w, h, n-1, m+1) - c;
                float gWN = getPixelF(FOTP, w, h, n-1, m-1) - c;

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
    // Edge pixels in FOTP = Fhat[edge] (correct boundary values, NOT zeroed)
}

// =============================================================================
// SSIM for CS fitness — comparing smooth pixels of FOTP vs Fhat
// =============================================================================
static float computeSSIM(const std::vector<float>& ref,
    const std::vector<float>& test,
    const std::vector<bool>& edgeMask,
    int w, int h)
{
    double mu1 = 0, mu2 = 0;
    int cnt = 0;
    for (size_t i = 0; i < (size_t)w*h; ++i) {
        if (edgeMask[i]) continue;
        mu1 += ref[i];
        mu2 += test[i];
        ++cnt;
    }
    if (cnt == 0) return 1.0f;
    mu1 /= cnt; mu2 /= cnt;

    double s1 = 0, s2 = 0, s12 = 0;
    for (size_t i = 0; i < (size_t)w*h; ++i) {
        if (edgeMask[i]) continue;
        double d1 = ref[i]-mu1, d2 = test[i]-mu2;
        s1 += d1*d1; s2 += d2*d2; s12 += d1*d2;
    }
    s1 /= cnt; s2 /= cnt; s12 /= cnt;

    const double C1 = (0.01*255)*(0.01*255);
    const double C2 = (0.03*255)*(0.03*255);
    double num = (2*mu1*mu2+C1)*(2*s12+C2);
    double den = (mu1*mu1+mu2*mu2+C1)*(s1+s2+C2);
    return (float)(num/den);
}

// =============================================================================
// CUCKOO SEARCH (Algorithm 2)
// =============================================================================
static float levyStep(std::mt19937& rng, float beta) {
    std::normal_distribution<float> nd(0, 1);
    double num = std::tgamma(1.0+beta)*std::sin(kPi*beta/2.0);
    double den = std::tgamma((1.0+beta)/2.0)*beta*std::pow(2.0,(beta-1.0)/2.0);
    float sig = (float)std::pow(std::abs(num/den), 1.0/beta);
    float u = nd(rng)*sig, v = nd(rng);
    if (std::abs(v) < 1e-10f) v = 1e-10f;
    return u / std::pow(std::abs(v), 1.0f/beta);
}

static float findLambda(const std::vector<float>& Fhat,
    const std::vector<bool>& edgeMask,
    int w, int h, const OdasParams& params)
{
    const float lMin = 0.0f, lMax = kPi/4.0f;
    const float ssimThreshold = 0.998f;
    const int n = params.csNests, MAXit = params.csMaxIter;
    const float pa = params.csPa, beta = params.csLevyBeta;

    auto fitness = [&](float lambda) -> float {
        if (lambda < 1e-6f) return 0.0f;
        std::vector<float> FOTP;
        applyODAD(Fhat, edgeMask, w, h, FOTP, lambda,
                  params.K, params.odadIterations);
        float ssim = computeSSIM(Fhat, FOTP, edgeMask, w, h);
        if (ssim < ssimThreshold) return -1.0f + ssim;
        return lambda * ssim;
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ur(lMin, lMax);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    std::vector<float> nests(n), fits(n);
    for (int i = 0; i < n; ++i) {
        nests[i] = ur(rng);
        fits[i]  = fitness(nests[i]);
    }

    int bi = (int)(std::max_element(fits.begin(),fits.end())-fits.begin());
    float bL = nests[bi], bF = fits[bi];

    std::cout << "  [ODAS-CS] n=" << n << " MAXit=" << MAXit
              << " λ_range=[0, π/4=" << std::fixed << std::setprecision(4)
              << lMax << "]" << std::endl;

    for (int t = 0; t < MAXit; ++t) {
        for (int i = 0; i < n; ++i) {
            float newL = nests[i] + 0.01f*(lMax-lMin)*levyStep(rng,beta);
            newL = std::max(lMin, std::min(lMax, newL));
            float newF = fitness(newL);
            int j = (int)(u01(rng)*n)%n;
            if (newF > fits[j]) {
                nests[j]=newL; fits[j]=newF;
                if (newF > bF) { bF=newF; bL=newL; }
            }
        }
        int na = std::max(1,(int)(pa*n));
        std::vector<int> si(n);
        std::iota(si.begin(),si.end(),0);
        std::sort(si.begin(),si.end(),[&](int a,int b){return fits[a]<fits[b];});
        for (int k = 0; k < na; ++k) {
            int idx=si[k];
            nests[idx]=ur(rng); fits[idx]=fitness(nests[idx]);
            if (fits[idx]>bF){bF=fits[idx];bL=nests[idx];}
        }
        if ((t+1)%10==0||t==MAXit-1)
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
//
// F_AES[edge]  = Laplacian HF response (additive sharpening delta)
// F_AES[smooth]= 0
// F_OTP[edge]  = Fhat[edge]  (fixed boundary, unchanged by ODAD)
// F_OTP[smooth]= ODAD-filtered value
//
// Result:
//   F_IHR[edge]  = Fhat[edge] + Laplacian_HF  → sharpened edge pixel
//   F_IHR[smooth]= ODAD_value + 0             → texture-preserved smooth pixel
// =============================================================================
static void composeIHR(const std::vector<float>& FAES,
    const std::vector<float>& FOTP,
    int w, int h, std::vector<float>& FIHR)
{
    size_t N = (size_t)w*h;
    FIHR.resize(N);
    for (size_t i = 0; i < N; ++i)
        FIHR[i] = std::max(0.0f, std::min(255.0f, FAES[i] + FOTP[i]));
}

// =============================================================================
// RESIDUAL SHARPENING (Eqs. 13-14)
// =============================================================================
static void residual(const std::vector<float>& FIHR,
    int w, int h, int sf, std::vector<float>& FRHR)
{
    int uw = w*sf, uh = h*sf;
    std::vector<float> up, bhr;
    L3up(FIHR, w, h, up, uw, uh);
    L3dn(up, uw, uh, bhr, w, h);
    size_t N = (size_t)w*h;
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
    size_t N=(size_t)w*h; Y.resize(N); Cb.resize(N); Cr.resize(N);
    for (size_t i = 0; i < N; ++i) {
        float r=rgb[i*4+0], g=rgb[i*4+1], b=rgb[i*4+2];
        Y[i]  =  0.299f*r + 0.587f*g + 0.114f*b;
        Cb[i] = -0.16874f*r - 0.33126f*g + 0.5f*b + 128.0f;
        Cr[i] =  0.5f*r - 0.41869f*g - 0.08131f*b + 128.0f;
    }
}
static void bilinUp(const std::vector<float>& src, int iw, int ih,
    std::vector<float>& dst, int ow, int oh)
{
    dst.resize((size_t)ow*oh);
    float sx=(float)iw/ow, sy=(float)ih/oh;
    for (int oy=0;oy<oh;++oy) {
        float fy=((float)oy+0.5f)*sy-0.5f; int y0=(int)std::floor(fy); float v=fy-y0;
        for (int ox=0;ox<ow;++ox) {
            float fx=((float)ox+0.5f)*sx-0.5f; int x0=(int)std::floor(fx); float u=fx-x0;
            dst[(size_t)oy*ow+ox]=
                (getPixelF(src,iw,ih,x0,y0)+u*(getPixelF(src,iw,ih,x0+1,y0)-getPixelF(src,iw,ih,x0,y0)))*(1-v)+
                (getPixelF(src,iw,ih,x0,y0+1)+u*(getPixelF(src,iw,ih,x0+1,y0+1)-getPixelF(src,iw,ih,x0,y0+1)))*v;
        }
    }
}
static void bilinUpRGBA(const unsigned char* src, int iw, int ih,
    unsigned char* dst, int ow, int oh)
{
    float sx=(float)iw/ow, sy=(float)ih/oh;
    for (int oy=0;oy<oh;++oy) {
        float fy=((float)oy+0.5f)*sy-0.5f;
        int y0=std::max(0,std::min(ih-1,(int)std::floor(fy)));
        int y1=std::max(0,std::min(ih-1,y0+1)); float v=fy-std::floor(fy);
        for (int ox=0;ox<ow;++ox) {
            float fx=((float)ox+0.5f)*sx-0.5f;
            int x0=std::max(0,std::min(iw-1,(int)std::floor(fx)));
            int x1=std::max(0,std::min(iw-1,x0+1)); float u=fx-std::floor(fx);
            for (int c=0;c<4;++c) {
                float p00=src[(y0*iw+x0)*4+c], p10=src[(y0*iw+x1)*4+c];
                float p01=src[(y1*iw+x0)*4+c], p11=src[(y1*iw+x1)*4+c];
                dst[(oy*ow+ox)*4+c]=clampByte((p00+u*(p10-p00))*(1-v)+(p01+u*(p11-p01))*v);
            }
        }
    }
}
static void ycbcr2rgba(const std::vector<float>& Y,
    const std::vector<float>& Cb, const std::vector<float>& Cr,
    const unsigned char* al, unsigned char* dst, int w, int h)
{
    for (size_t i=0;i<(size_t)w*h;++i) {
        float y=Y[i], cb=Cb[i]-128.0f, cr=Cr[i]-128.0f;
        dst[i*4+0]=clampByte(y+1.402f*cr);
        dst[i*4+1]=clampByte(y-0.34414f*cb-0.71414f*cr);
        dst[i*4+2]=clampByte(y+1.772f*cb);
        dst[i*4+3]=al[i*4+3];
    }
}

// =============================================================================
// PIPELINE
// =============================================================================
static std::vector<float> odasPipeline(
    const std::vector<float>& ch,
    int inW, int inH, int outW, int outH, int sf,
    const OdasParams& params, bool verbose)
{
    // Step 1: Lanczos3 upscale
    if (verbose) std::cout<<"  [ODAS] Step 1: Lanczos3 upscale..."<<std::endl;
    std::vector<float> Fhat;
    L3up(ch, inW, inH, Fhat, outW, outH);

    // Step 2: Canny edge mask
    if (verbose) std::cout<<"  [ODAS] Step 2: Canny edge detection..."<<std::endl;
    std::vector<bool> edgeMask;
    cannyEdgeMask(Fhat, outW, outH, edgeMask,
                  params.cannyLowThresh, params.cannyHighThresh);
    if (verbose) {
        size_t ec=std::count(edgeMask.begin(),edgeMask.end(),true);
        std::cout<<"  [ODAS] Edge pixels: "<<ec<<"/"<<(size_t)outW*outH
                 <<" ("<<std::fixed<<std::setprecision(2)
                 <<100.0f*ec/(outW*outH)<<"%)"<<std::endl;
    }

    // Step 3: AES — Laplacian HF response at edge pixels
    if (verbose) std::cout<<"  [ODAS] Step 3: AES (Laplacian HF at edges)..."<<std::endl;
    std::vector<float> FAES;
    applyAES(Fhat, edgeMask, outW, outH, FAES);
    if (verbose) {
        double chg=0; int cnt=0;
        for (size_t i=0;i<(size_t)outW*outH;++i)
            if (edgeMask[i]){chg+=std::abs(FAES[i]);++cnt;}
        std::cout<<"  [ODAS] AES mean |HF response|: "
                 <<(cnt>0?chg/cnt:0.0)<<" grey levels"<<std::endl;
    }

    // Step 4: CS for lambda
    if (verbose) std::cout<<"  [ODAS] Step 4: Cuckoo Search for λ..."<<std::endl;
    float lambda = findLambda(Fhat, edgeMask, outW, outH, params);

    // Step 5: ODAD — smooth region diffusion, edges fixed at Fhat
    if (verbose)
        std::cout<<"  [ODAS] Step 5: ODAD (λ="<<std::fixed<<std::setprecision(4)
                 <<lambda<<", K="<<params.K<<")..."<<std::endl;
    std::vector<float> FOTP;
    applyODAD(Fhat, edgeMask, outW, outH, FOTP, lambda,
              params.K, params.odadIterations);
    if (verbose) {
        double chg=0; int cnt=0;
        for (size_t i=0;i<(size_t)outW*outH;++i)
            if (!edgeMask[i]){chg+=std::abs(FOTP[i]-Fhat[i]);++cnt;}
        std::cout<<"  [ODAS] ODAD mean smooth change: "
                 <<(cnt>0?chg/cnt:0.0)<<" grey levels"<<std::endl;
    }

    // Step 6: F_IHR = F_AES + F_OTP
    // F_AES[edge]=HF_delta, F_AES[smooth]=0
    // F_OTP[edge]=Fhat[edge], F_OTP[smooth]=ODAD_value
    // -> F_IHR[edge]=Fhat+HF_delta (sharpened), F_IHR[smooth]=ODAD_value
    if (verbose) std::cout<<"  [ODAS] Step 6: IHR = F_AES + F_OTP..."<<std::endl;
    std::vector<float> FIHR;
    composeIHR(FAES, FOTP, outW, outH, FIHR);

    // Step 7: Residual
    if (verbose) std::cout<<"  [ODAS] Step 7: Residual sharpening..."<<std::endl;
    std::vector<float> FRHR;
    residual(FIHR, outW, outH, sf, FRHR);
    if (verbose) {
        double chg=0;
        for (size_t i=0;i<(size_t)outW*outH;++i)
            chg+=std::abs(FRHR[i]-FIHR[i]);
        std::cout<<"  [ODAS] Residual mean change: "
                 <<chg/(outW*outH)<<" grey levels"<<std::endl;
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
    int sf = std::max(1,(int)std::round((float)outW/inW));
    std::cout<<"  [ODAS] Input:  "<<inW<<"x"<<inH<<std::endl;
    std::cout<<"  [ODAS] Output: "<<outW<<"x"<<outH<<std::endl;
    std::cout<<"  [ODAS] Scale:  ~"<<sf<<"x  K="<<params.K
             <<"  iters="<<params.odadIterations<<std::endl;

    std::vector<unsigned char> bilin((size_t)outW*outH*4);
    bilinUpRGBA(input, inW, inH, bilin.data(), outW, outH);

    if (params.useYCbCr) {
        std::vector<float> Y, Cb, Cr;
        rgb2ycbcr(input, inW, inH, Y, Cb, Cr);
        auto FRHR = odasPipeline(Y, inW, inH, outW, outH, sf, params, true);
        std::vector<float> CbO, CrO;
        bilinUp(Cb, inW, inH, CbO, outW, outH);
        bilinUp(Cr, inW, inH, CrO, outW, outH);
        std::cout<<"  [ODAS] Step 8: RGBA output..."<<std::endl;
        ycbcr2rgba(FRHR, CbO, CrO, bilin.data(), output, outW, outH);
    } else {
        std::cout<<"  [ODAS] RGB mode"<<std::endl;
        std::vector<std::vector<float>> ch(3);
        for (int c=0;c<3;++c) {
            std::vector<float> ci((size_t)inW*inH);
            for (int i=0;i<inW*inH;++i) ci[i]=input[i*4+c];
            ch[c]=odasPipeline(ci, inW, inH, outW, outH, sf, params, c==0);
        }
        for (int i=0;i<outW*outH;++i) {
            output[i*4+0]=clampByte(ch[0][i]);
            output[i*4+1]=clampByte(ch[1][i]);
            output[i*4+2]=clampByte(ch[2][i]);
            output[i*4+3]=bilin[i*4+3];
        }
    }
    std::cout<<"  [ODAS] Done."<<std::endl;
}
