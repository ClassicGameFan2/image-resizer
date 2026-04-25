// =============================================================================
// odas.cpp  — v3: Fixed boundary conditions, fixed CS fitness
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

static constexpr float kPi = 3.14159265358979323846f;

static inline unsigned char clampByte(float v) {
    return (unsigned char)std::max(0.0f, std::min(255.0f, v + 0.5f));
}

// Mirror padding fetch
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
// SECTION 1: Lanczos3 (unchanged from v2)
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
    const std::vector<float>& src, int inW, int inH,
    std::vector<float>&       dst, int outW, int outH)
{
    dst.resize((size_t)outW * outH);
    float scaleX = (float)inW / (float)outW;
    float scaleY = (float)inH / (float)outH;

    for (int oy = 0; oy < outH; ++oy) {
        float srcY = ((float)oy + 0.5f) * scaleY - 0.5f;
        int   cy   = (int)std::floor(srcY);
        for (int ox = 0; ox < outW; ++ox) {
            float srcX = ((float)ox + 0.5f) * scaleX - 0.5f;
            int   cx   = (int)std::floor(srcX);
            float sum = 0.0f, wSum = 0.0f;
            for (int ky = cy - 2; ky <= cy + 3; ++ky) {
                float wy = lanczos3Weight(srcY - (float)ky);
                for (int kx = cx - 2; kx <= cx + 3; ++kx) {
                    float w = lanczos3Weight(srcX - (float)kx) * wy;
                    sum  += getPixelF(src, inW, inH, kx, ky) * w;
                    wSum += w;
                }
            }
            float val = (wSum > 1e-7f) ? sum / wSum
                                       : getPixelF(src, inW, inH, cx, cy);
            dst[(size_t)oy * outW + ox] = std::max(0.0f, std::min(255.0f, val));
        }
    }
}

static void lanczos3Downscale(
    const std::vector<float>& src, int inW, int inH,
    std::vector<float>&       dst, int outW, int outH)
{
    dst.resize((size_t)outW * outH);
    float scaleX = (float)inW / (float)outW;
    float scaleY = (float)inH / (float)outH;
    float radX   = 3.0f * scaleX;
    float radY   = 3.0f * scaleY;

    for (int oy = 0; oy < outH; ++oy) {
        float srcY = ((float)oy + 0.5f) * scaleY - 0.5f;
        int yMin = (int)std::floor(srcY - radY + 1);
        int yMax = (int)std::floor(srcY + radY);
        for (int ox = 0; ox < outW; ++ox) {
            float srcX = ((float)ox + 0.5f) * scaleX - 0.5f;
            int xMin = (int)std::floor(srcX - radX + 1);
            int xMax = (int)std::floor(srcX + radX);
            float sum = 0.0f, wSum = 0.0f;
            for (int ky = yMin; ky <= yMax; ++ky) {
                float wy = lanczos3Weight((srcY - (float)ky) / scaleY);
                for (int kx = xMin; kx <= xMax; ++kx) {
                    float w = lanczos3Weight((srcX - (float)kx) / scaleX) * wy;
                    sum  += getPixelF(src, inW, inH, kx, ky) * w;
                    wSum += w;
                }
            }
            int cy = std::max(0, std::min(inH-1, (int)srcY));
            int cx = std::max(0, std::min(inW-1, (int)srcX));
            float val = (wSum > 1e-7f) ? sum / wSum
                                       : getPixelF(src, inW, inH, cx, cy);
            dst[(size_t)oy * outW + ox] = std::max(0.0f, std::min(255.0f, val));
        }
    }
}

// =============================================================================
// SECTION 2: Canny Edge Detection (unchanged from v2)
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
    dst.resize(src.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int ky = -2; ky <= 2; ++ky)
                for (int kx = -2; kx <= 2; ++kx)
                    acc += K[ky+2][kx+2] * getPixelF(src, w, h, x+kx, y+ky);
            dst[(size_t)y * w + x] = acc / 273.0f;
        }
}

static void cannyEdgeDetect(
    const std::vector<float>& img, int w, int h,
    std::vector<bool>& edgeMask,
    float lowThresh, float highThresh)
{
    int N = w * h;
    std::vector<float> blurred;
    gaussianBlur5(img, w, h, blurred);

    std::vector<float> gradMag(N, 0.0f), gradDir(N, 0.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float gx =
                -1*getPixelF(blurred,w,h,x-1,y-1) + 1*getPixelF(blurred,w,h,x+1,y-1) +
                -2*getPixelF(blurred,w,h,x-1,y  ) + 2*getPixelF(blurred,w,h,x+1,y  ) +
                -1*getPixelF(blurred,w,h,x-1,y+1) + 1*getPixelF(blurred,w,h,x+1,y+1);
            float gy =
                -1*getPixelF(blurred,w,h,x-1,y-1) - 2*getPixelF(blurred,w,h,x,y-1) +
                -1*getPixelF(blurred,w,h,x+1,y-1) + 1*getPixelF(blurred,w,h,x-1,y+1) +
                 2*getPixelF(blurred,w,h,x,  y+1) + 1*getPixelF(blurred,w,h,x+1,y+1);
            size_t idx = (size_t)y*w+x;
            gradMag[idx] = std::sqrt(gx*gx + gy*gy);
            gradDir[idx] = std::atan2(gy, gx);
        }

    std::vector<float> suppressed(N, 0.0f);
    for (int y = 1; y < h-1; ++y)
        for (int x = 1; x < w-1; ++x) {
            size_t idx = (size_t)y*w+x;
            float mag = gradMag[idx];
            float angle = gradDir[idx] * 180.0f / kPi;
            if (angle < 0) angle += 180.0f;
            float m1, m2;
            if      (angle < 22.5f || angle >= 157.5f) { m1=gradMag[idx+1];       m2=gradMag[idx-1]; }
            else if (angle < 67.5f)                    { m1=gradMag[idx-(size_t)w+1]; m2=gradMag[idx+(size_t)w-1]; }
            else if (angle < 112.5f)                   { m1=gradMag[idx-(size_t)w];   m2=gradMag[idx+(size_t)w]; }
            else                                       { m1=gradMag[idx-(size_t)w-1]; m2=gradMag[idx+(size_t)w+1]; }
            suppressed[idx] = (mag >= m1 && mag >= m2) ? mag : 0.0f;
        }

    std::vector<int> status(N, 0);
    for (int i = 0; i < N; ++i) {
        if      (suppressed[i] >= highThresh) status[i] = 2;
        else if (suppressed[i] >= lowThresh)  status[i] = 1;
    }
    std::vector<int> stack;
    stack.reserve(N/8);
    for (int i = 0; i < N; ++i) if (status[i]==2) stack.push_back(i);
    while (!stack.empty()) {
        int idx = stack.back(); stack.pop_back();
        int px = idx%w, py = idx/w;
        for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
            if (!dx&&!dy) continue;
            int nx=px+dx, ny=py+dy;
            if (nx<0||nx>=w||ny<0||ny>=h) continue;
            int ni=ny*w+nx;
            if (status[ni]==1) { status[ni]=2; stack.push_back(ni); }
        }
    }
    edgeMask.assign(N, false);
    for (int i = 0; i < N; ++i) edgeMask[i] = (status[i]==2);
}

// =============================================================================
// SECTION 3: AES - unchanged from v2 (was correct)
// =============================================================================

static void applyAES(
    const std::vector<float>& Fhat,
    const std::vector<bool>&  edgeMask,
    int w, int h,
    std::vector<float>&       FAES)
{
    const size_t N = (size_t)w * h;
    FAES.resize(N);

    std::vector<float> localVar(N, 0.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            size_t idx = (size_t)y*w+x;
            if (!edgeMask[idx]) continue;
            float s = 0;
            for (int ky=-1;ky<=1;++ky) for (int kx=-1;kx<=1;++kx)
                s += getPixelF(Fhat,w,h,x+kx,y+ky);
            float mean = s / 9.0f, v = 0;
            for (int ky=-1;ky<=1;++ky) for (int kx=-1;kx<=1;++kx) {
                float d = getPixelF(Fhat,w,h,x+kx,y+ky) - mean;
                v += d*d;
            }
            localVar[idx] = v / 9.0f;
        }

    float VRmin = 1e30f, VRmax = -1e30f;
    for (size_t i = 0; i < N; ++i) {
        if (!edgeMask[i]) continue;
        VRmin = std::min(VRmin, localVar[i]);
        VRmax = std::max(VRmax, localVar[i]);
    }
    if (VRmin > VRmax) { FAES = Fhat; return; }

    float S = (VRmax - VRmin) / 4.0f;
    if (S < 1e-6f) S = 1.0f;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t idx = (size_t)y*w+x;
            if (!edgeMask[idx]) {
                FAES[idx] = Fhat[idx];
                continue;
            }
            float lvr = localVar[idx];
            float cmp;
            if      (lvr <= VRmin +       S) cmp = 16.0f;
            else if (lvr <= VRmin + 2.0f*S)  cmp = 24.0f;
            else if (lvr <= VRmin + 3.0f*S)  cmp = 32.0f;
            else if (lvr <= VRmin + 4.0f*S)  cmp = 40.0f;
            else                              cmp = 48.0f;

            float nw = -cmp / 8.0f;
            float result = cmp * getPixelF(Fhat,w,h,x,y);
            for (int ky=-1;ky<=1;++ky) for (int kx=-1;kx<=1;++kx) {
                if (!ky&&!kx) continue;
                result += nw * getPixelF(Fhat,w,h,x+kx,y+ky);
            }
            FAES[idx] = std::max(0.0f, std::min(255.0f, result));
        }
    }
}

// =============================================================================
// SECTION 4: ODAD Filter — v3 FIX: Fixed boundary conditions
//
// THE CORE BUG IN v1 AND v2:
//   F_s had zeros at edge pixel locations.
//   ODAD diffusion read these zeros as neighbors of smooth pixels.
//   Smooth pixels near edges were pulled toward zero → black lines.
//
// THE FIX:
//   ODAD operates on the FULL F_hat image.
//   Edge pixels are treated as FIXED BOUNDARY CONDITIONS — they are
//   read by the diffusion but never updated themselves.
//   Only smooth (non-edge) pixels are updated by the diffusion equation.
//
//   This is mathematically correct: edges are anchors, smooth regions
//   diffuse between them. No artificial zeros pollute the diffusion.
// =============================================================================

static void applyODAD(
    const std::vector<float>& Fhat,     // FULL image (not sparse F_s)
    const std::vector<bool>&  edgeMask, // true = fixed boundary pixel
    int w, int h,
    std::vector<float>&       FOTP,
    float lambda, float K, int iterations)
{
    const size_t N = (size_t)w * h;
    const float  K2 = K * K;

    // Start from F_hat — edge pixels stay fixed, smooth pixels diffuse
    FOTP = Fhat;

    for (int t = 0; t < iterations; ++t) {
        std::vector<float> next = FOTP;  // copy: edge pixels unchanged

        for (int m = 0; m < h; ++m) {
            for (int n = 0; n < w; ++n) {
                size_t idx = (size_t)m * w + n;

                // Edge pixels are fixed boundary conditions — never updated
                if (edgeMask[idx]) continue;

                float center = FOTP[idx];

                // 8 directional gradients — read from full FOTP including
                // fixed edge pixels. This is the correct boundary condition.
                float gN  = getPixelF(FOTP,w,h,n,  m-1) - center;
                float gS  = getPixelF(FOTP,w,h,n,  m+1) - center;
                float gE  = getPixelF(FOTP,w,h,n+1,m  ) - center;
                float gW  = getPixelF(FOTP,w,h,n-1,m  ) - center;
                float gNE = getPixelF(FOTP,w,h,n+1,m-1) - center;
                float gSE = getPixelF(FOTP,w,h,n+1,m+1) - center;
                float gWS = getPixelF(FOTP,w,h,n-1,m+1) - center;
                float gWN = getPixelF(FOTP,w,h,n-1,m-1) - center;

                auto dc = [&](float g) {
                    return std::exp(-(g * g) / K2);
                };

                float update = lambda * (
                    dc(gN)*gN  + dc(gS)*gS  + dc(gE)*gE  + dc(gW)*gW +
                    dc(gNE)*gNE + dc(gSE)*gSE + dc(gWS)*gWS + dc(gWN)*gWN
                );

                next[idx] = std::max(0.0f, std::min(255.0f, center + update));
            }
        }
        FOTP = next;
    }
}

// =============================================================================
// SECTION 5: Cuckoo Search — v3 FIX: Correct fitness function
//
// PROBLEM WITH SSIM:
//   In a typical photo ~97% of pixels are smooth. The ODAD filter with
//   t=1 and small λ barely changes smooth pixels. SSIM between two nearly-
//   identical smooth images is always ≈1.0 regardless of λ.
//
// SOLUTION: Laplacian Energy fitness
//   Measure the sum of squared Laplacian responses in the smooth region
//   of the ODAD output. Higher energy = more texture detail preserved.
//   The Laplacian measures local contrast/texture — exactly what we want
//   the ODAD filter to preserve.
//
//   fitness(λ) = Σ_{smooth pixels} (Laplacian(ODAD(Fhat,λ))[i])²
//
//   A good λ:
//     - Too small: barely filters → same energy as F_hat (OK but no benefit)
//     - Too large: over-smooths → energy drops → bad fitness
//     - Optimal: maximally preserves texture energy while diffusing noise
//
// NOTE: Since t=1 and K=0.2, the filter effect is subtle. The CS will
// find the λ that maximally preserves local texture contrast.
// =============================================================================

// Compute Laplacian energy of smooth pixels in an image
static float laplacianEnergy(
    const std::vector<float>& img,
    const std::vector<bool>&  edgeMask,
    int w, int h)
{
    float energy = 0.0f;
    int   count  = 0;

    for (int y = 1; y < h-1; ++y) {
        for (int x = 1; x < w-1; ++x) {
            size_t idx = (size_t)y * w + x;
            if (edgeMask[idx]) continue;  // skip edge pixels

            // 4-connected discrete Laplacian
            float lap =
                img[idx - w] + img[idx + w] +
                img[idx - 1] + img[idx + 1] -
                4.0f * img[idx];

            energy += lap * lap;
            ++count;
        }
    }
    return (count > 0) ? (energy / (float)count) : 0.0f;
}

static float levyStep(std::mt19937& rng, float beta) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float num   = std::tgamma(1.0f + beta) * std::sin(kPi * beta / 2.0f);
    float den   = std::tgamma((1.0f + beta) / 2.0f) * beta *
                  std::pow(2.0f, (beta - 1.0f) / 2.0f);
    float sigma = std::pow(std::abs(num / den), 1.0f / beta);
    float u     = nd(rng) * sigma;
    float v     = nd(rng);
    if (std::abs(v) < 1e-10f) v = 1e-10f;
    return u / std::pow(std::abs(v), 1.0f / beta);
}

static float cuckooSearchLambda(
    const std::vector<float>& Fhat,
    const std::vector<bool>&  edgeMask,
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

    // Fitness: Laplacian energy of smooth region in ODAD output.
    // Higher = more texture detail preserved = better.
    auto fitness = [&](float lambda) -> float {
        std::vector<float> filtered;
        applyODAD(Fhat, edgeMask, w, h, filtered,
                  lambda, params.K, params.odadIterations);
        return laplacianEnergy(filtered, edgeMask, w, h);
    };

    // Initialize population
    std::vector<float> nests(n);
    std::vector<float> fitnesses(n);
    for (int i = 0; i < n; ++i) {
        nests[i]     = uniRange(rng);
        fitnesses[i] = fitness(nests[i]);
    }

    int   bestIdx    = (int)(std::max_element(fitnesses.begin(),
                                               fitnesses.end())
                             - fitnesses.begin());
    float bestLambda = nests[bestIdx];
    float bestFit    = fitnesses[bestIdx];

    std::cout << "  [ODAS-CS] Starting Cuckoo Search for optimal λ ∈ [0, π/4]"
              << std::endl;
    std::cout << "  [ODAS-CS] Population=" << n << "  MaxIter=" << MAXit
              << "  Pa=" << pa << std::endl;

    for (int t = 0; t < MAXit; ++t) {
        float stepScale = 0.01f * (lambdaMax - lambdaMin) /
                          (1.0f + (float)t * 0.05f);

        for (int i = 0; i < n; ++i) {
            float newLambda = std::max(lambdaMin,
                              std::min(lambdaMax,
                                       nests[i] + stepScale * levyStep(rng, beta)));
            float newFit    = fitness(newLambda);

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

        // Abandon worst nests
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
                      << "  Energy=" << bestFit << std::endl;
        }
    }

    std::cout << "  [ODAS-CS] Optimal λ = " << bestLambda
              << "  (Energy=" << bestFit << ")" << std::endl;
    return bestLambda;
}

// =============================================================================
// SECTION 6: IHR Composition — v3 FIX: correct masked merge
//
// FOTP now contains the full image (Fhat values at edge pixels,
// diffused values at smooth pixels). F_AES has sharpened values at
// edge pixels, Fhat values at smooth pixels.
//
// Correct composition:
//   Edge pixels    → F_AES (adaptively sharpened)
//   Smooth pixels  → F_OTP (ODAD diffused with proper boundary conditions)
// =============================================================================

static void composeIHR(
    const std::vector<float>& FAES,
    const std::vector<float>& FOTP,
    const std::vector<bool>&  edgeMask,
    int w, int h,
    std::vector<float>&       FIHR)
{
    const size_t N = (size_t)w * h;
    FIHR.resize(N);
    for (size_t i = 0; i < N; ++i)
        FIHR[i] = std::max(0.0f, std::min(255.0f,
                      edgeMask[i] ? FAES[i] : FOTP[i]));
}

// =============================================================================
// SECTION 7: Residual Sharpening (unchanged)
// =============================================================================

static void residualSharpening(
    const std::vector<float>& FIHR, int w, int h,
    int scaleFactor,
    std::vector<float>&       FRHR)
{
    int upW = w * scaleFactor, upH = h * scaleFactor;
    std::vector<float> upscaled, FBHR;
    lanczos3Upscale(FIHR, w, h, upscaled, upW, upH);
    lanczos3Downscale(upscaled, upW, upH, FBHR, w, h);

    const size_t N = (size_t)w * h;
    FRHR.resize(N);
    for (size_t i = 0; i < N; ++i)
        FRHR[i] = std::max(0.0f, std::min(255.0f,
                      FIHR[i] + (FIHR[i] - FBHR[i])));
}

// =============================================================================
// SECTION 8: Color space helpers (unchanged from v2)
// =============================================================================

static void rgbToYCbCr(
    const unsigned char* rgb, int w, int h,
    std::vector<float>& Y, std::vector<float>& Cb, std::vector<float>& Cr)
{
    const size_t N = (size_t)w * h;
    Y.resize(N); Cb.resize(N); Cr.resize(N);
    for (size_t i = 0; i < N; ++i) {
        float r=rgb[i*4],g=rgb[i*4+1],b=rgb[i*4+2];
        Y[i]  =  0.299f*r + 0.587f*g + 0.114f*b;
        Cb[i] = -0.16874f*r - 0.33126f*g + 0.5f*b + 128.0f;
        Cr[i] =  0.5f*r - 0.41869f*g - 0.08131f*b + 128.0f;
    }
}

static void bilinearUpscale(
    const std::vector<float>& src, int inW, int inH,
    std::vector<float>&       dst, int outW, int outH)
{
    dst.resize((size_t)outW * outH);
    float sx=(float)inW/(float)outW, sy=(float)inH/(float)outH;
    for (int oy=0;oy<outH;++oy) {
        float fy=((float)oy+0.5f)*sy-0.5f;
        int y0=(int)std::floor(fy); float v=fy-(float)y0;
        for (int ox=0;ox<outW;++ox) {
            float fx=((float)ox+0.5f)*sx-0.5f;
            int x0=(int)std::floor(fx); float u=fx-(float)x0;
            float p00=getPixelF(src,inW,inH,x0,  y0);
            float p10=getPixelF(src,inW,inH,x0+1,y0);
            float p01=getPixelF(src,inW,inH,x0,  y0+1);
            float p11=getPixelF(src,inW,inH,x0+1,y0+1);
            dst[(size_t)oy*outW+ox]=(p00+u*(p10-p00))*(1-v)+(p01+u*(p11-p01))*v;
        }
    }
}

static void bilinearUpscaleRGBA(
    const unsigned char* src, int inW, int inH,
    unsigned char* dst, int outW, int outH)
{
    float sx=(float)inW/(float)outW, sy=(float)inH/(float)outH;
    for (int oy=0;oy<outH;++oy) {
        float fy=((float)oy+0.5f)*sy-0.5f;
        int y0=std::max(0,std::min(inH-1,(int)std::floor(fy)));
        int y1=std::max(0,std::min(inH-1,y0+1));
        float v=fy-std::floor(fy);
        for (int ox=0;ox<outW;++ox) {
            float fx=((float)ox+0.5f)*sx-0.5f;
            int x0=std::max(0,std::min(inW-1,(int)std::floor(fx)));
            int x1=std::max(0,std::min(inW-1,x0+1));
            float u=fx-std::floor(fx);
            for (int c=0;c<4;++c) {
                float p00=src[(y0*inW+x0)*4+c], p10=src[(y0*inW+x1)*4+c];
                float p01=src[(y1*inW+x0)*4+c], p11=src[(y1*inW+x1)*4+c];
                dst[(oy*outW+ox)*4+c]=clampByte((p00+u*(p10-p00))*(1-v)
                                                +(p01+u*(p11-p01))*v);
            }
        }
    }
}

static void yCbCrToRgba(
    const std::vector<float>& Y,
    const std::vector<float>& Cb,
    const std::vector<float>& Cr,
    const unsigned char* alphaSource,
    unsigned char* dst, int w, int h)
{
    const size_t N = (size_t)w * h;
    for (size_t i = 0; i < N; ++i) {
        float y=Y[i], cb=Cb[i]-128.0f, cr=Cr[i]-128.0f;
        dst[i*4+0] = clampByte(y + 1.40200f*cr);
        dst[i*4+1] = clampByte(y - 0.34414f*cb - 0.71414f*cr);
        dst[i*4+2] = clampByte(y + 1.77200f*cb);
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

    std::vector<unsigned char> bilinearOut((size_t)outW * outH * 4);
    bilinearUpscaleRGBA(input, inW, inH, bilinearOut.data(), outW, outH);

    if (params.useYCbCr) {
        // ── YCbCr mode ────────────────────────────────────────────────────

        std::vector<float> Yin, Cbin, Crin;
        rgbToYCbCr(input, inW, inH, Yin, Cbin, Crin);

        // Step 1: Lanczos3 upscale Y → F_hat
        std::cout << "  [ODAS] Step 1: Lanczos3 upscale..." << std::endl;
        std::vector<float> Fhat;
        lanczos3Upscale(Yin, inW, inH, Fhat, outW, outH);

        // Step 2: Canny edge detection
        std::cout << "  [ODAS] Step 2: Canny edge detection..." << std::endl;
        std::vector<bool> edgeMask;
        cannyEdgeDetect(Fhat, outW, outH, edgeMask,
                        params.cannyLowThresh, params.cannyHighThresh);

        size_t edgeCount = std::count(edgeMask.begin(), edgeMask.end(), true);
        std::cout << "  [ODAS] Edge pixels: " << edgeCount
                  << " / " << ((size_t)outW * outH)
                  << " (" << (100.0f * edgeCount / (outW * outH)) << "%)"
                  << std::endl;

        // Step 3: AES on F_hat at edge locations → F_AES (complete image)
        std::cout << "  [ODAS] Step 3: Adaptive Edge Sharpening..." << std::endl;
        std::vector<float> FAES;
        applyAES(Fhat, edgeMask, outW, outH, FAES);

        // Step 4: Cuckoo Search for optimal λ using Laplacian energy
        std::cout << "  [ODAS] Step 4: Cuckoo Search optimization for λ..."
                  << std::endl;
        float lambda = cuckooSearchLambda(Fhat, edgeMask, outW, outH, params);

        // Step 5: ODAD filter on F_hat with edge pixels fixed → F_OTP
        std::cout << "  [ODAS] Step 5: ODAD filter (λ=" << lambda << ")..."
                  << std::endl;
        std::vector<float> FOTP;
        applyODAD(Fhat, edgeMask, outW, outH, FOTP,
                  lambda, params.K, params.odadIterations);

        // Step 6: IHR masked merge
        std::cout << "  [ODAS] Step 6: IHR composition..." << std::endl;
        std::vector<float> FIHR;
        composeIHR(FAES, FOTP, edgeMask, outW, outH, FIHR);

        // Step 7: Residual sharpening
        std::cout << "  [ODAS] Step 7: Residual-based sharpening..." << std::endl;
        std::vector<float> FRHR;
        residualSharpening(FIHR, outW, outH, scaleFactor, FRHR);

        // Step 8: Bilinear upscale Cb, Cr
        std::vector<float> CbOut, CrOut;
        bilinearUpscale(Cbin, inW, inH, CbOut, outW, outH);
        bilinearUpscale(Crin, inW, inH, CrOut, outW, outH);

        // Step 9: YCbCr → RGBA
        std::cout << "  [ODAS] Step 9: Compositing final RGBA output..."
                  << std::endl;
        yCbCrToRgba(FRHR, CbOut, CrOut, bilinearOut.data(), output, outW, outH);

    } else {
        // ── RGB mode ──────────────────────────────────────────────────────
        std::cout << "  [ODAS] RGB mode: 3 channels independently" << std::endl;

        std::vector<std::vector<float>> channelFinal(3);
        for (int ch = 0; ch < 3; ++ch) {
            std::cout << "  [ODAS] Channel " << ch+1 << "/3..." << std::endl;

            std::vector<float> chanIn((size_t)inW * inH);
            for (int i = 0; i < inW * inH; ++i)
                chanIn[i] = (float)input[i*4+ch];

            std::vector<float> Fhat;
            lanczos3Upscale(chanIn, inW, inH, Fhat, outW, outH);

            std::vector<bool> edgeMask;
            cannyEdgeDetect(Fhat, outW, outH, edgeMask,
                            params.cannyLowThresh, params.cannyHighThresh);

            std::vector<float> FAES;
            applyAES(Fhat, edgeMask, outW, outH, FAES);

            float lambda = cuckooSearchLambda(Fhat, edgeMask, outW, outH, params);

            std::vector<float> FOTP;
            applyODAD(Fhat, edgeMask, outW, outH, FOTP,
                      lambda, params.K, params.odadIterations);

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
