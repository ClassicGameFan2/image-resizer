// =============================================================================
// ola_espline.cpp
// Implementation of OLA e-spline upscaling.
// Based on: "An improved Image Interpolation technique using OLA e-spline"
//            by Jagyanseni Panda and Sukadev Meher
// =============================================================================

#include "ola_espline.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <limits>
#include <iostream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Internal single-channel float image (row-major, x=column, y=row)
// Pixel values in [0, 255].
// =============================================================================
struct FImg
{
    int w = 0, h = 0;
    std::vector<float> d;

    FImg() = default;
    FImg(int w_, int h_) : w(w_), h(h_), d(w_ * h_, 0.f) {}

    float  at(int x, int y) const { return d[y * w + x]; }
    float& at(int x, int y)       { return d[y * w + x]; }

    float sample(int x, int y) const
    {
        x = std::max(0, std::min(x, w - 1));
        y = std::max(0, std::min(y, h - 1));
        return d[y * w + x];
    }

    // Diagnostic: compute min, max, mean of pixel values
    void stats(float& mn, float& mx, float& mean) const
    {
        mn = mx = d.empty() ? 0.f : d[0];
        double sum = 0.0;
        for (float v : d) {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        mean = d.empty() ? 0.f : (float)(sum / d.size());
    }
};

static void printImgStats(const char* label, const FImg& img)
{
    float mn, mx, mean;
    img.stats(mn, mx, mean);
    std::cout << "  [OLA diag] " << label
              << "  min=" << std::fixed << std::setprecision(2) << mn
              << "  max=" << mx
              << "  mean=" << mean
              << "  size=" << img.w << "x" << img.h
              << std::endl;
}

// =============================================================================
// SECTION 1 — LA GAUSSIAN FILTERING  [Algorithm 1]
// =============================================================================

static void localStats(const FImg& img, int cx, int cy,
                       float& mean, float& var)
{
    float sum = 0.f, sumSq = 0.f;
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
    {
        float v = img.sample(cx + dx, cy + dy);
        sum   += v;
        sumSq += v * v;
    }
    mean = sum / 9.f;
    var  = sumSq / 9.f - mean * mean;
    if (var < 0.f) var = 0.f;
}

static void globalVarianceBounds(const FImg& src,
                                  float& V_max, float& V_min)
{
    V_max = -1.f;
    V_min = std::numeric_limits<float>::max();
    for (int y = 0; y < src.h; ++y)
    for (int x = 0; x < src.w; ++x)
    {
        float m, v;
        localStats(src, x, y, m, v);
        if (v > V_max) V_max = v;
        if (v < V_min) V_min = v;
    }
}

static int selectCp(float v, float V_max, float S)
{
    if (S < 1e-9f) return 32;
    if      (v > V_max - 1.f * S) return 2;
    else if (v > V_max - 2.f * S) return 3;
    else if (v > V_max - 3.f * S) return 4;
    else if (v > V_max - 4.f * S) return 8;
    else if (v > V_max - 5.f * S) return 16;
    else                           return 32;
}

static const float kW[3][3] = {
    {1.f, 2.f, 1.f},
    {1.f, 0.f, 1.f},
    {1.f, 2.f, 1.f}
};

static FImg laAdaptiveGaussianBlur(const FImg& src)
{
    float V_max, V_min;
    globalVarianceBounds(src, V_max, V_min);

    float S = (V_max - V_min) / 6.f;

    std::cout << "  [OLA diag] LA filter: V_max=" << V_max
              << " V_min=" << V_min << " S=" << S << std::endl;

    // Count how many pixels get each Cp value
    int cpCount[6] = {0,0,0,0,0,0}; // for Cp = 2,3,4,8,16,32
    static const int cpVals[6] = {2,3,4,8,16,32};

    FImg dst(src.w, src.h);

    for (int y = 0; y < src.h; ++y)
    for (int x = 0; x < src.w; ++x)
    {
        float m, v;
        localStats(src, x, y, m, v);

        int Cp   = selectCp(v, V_max, S);
        float norm = static_cast<float>(Cp) + 12.f;

        float acc = static_cast<float>(Cp) * src.at(x, y);
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            acc += kW[dy + 1][dx + 1] * src.sample(x + dx, y + dy);
        }
        dst.at(x, y) = acc / norm;

        for (int ci = 0; ci < 6; ++ci)
            if (Cp == cpVals[ci]) { cpCount[ci]++; break; }
    }

    std::cout << "  [OLA diag] Cp distribution: ";
    for (int ci = 0; ci < 6; ++ci)
        std::cout << "Cp=" << cpVals[ci] << ":" << cpCount[ci] << " ";
    std::cout << std::endl;

    return dst;
}

// =============================================================================
// SECTION 2 — HPF + USM  [Eq. 4, 1]
// =============================================================================

static FImg computeHPF(const FImg& G_LR, const FImg& H_Ab)
{
    FImg H(G_LR.w, G_LR.h);
    for (int i = 0; i < (int)G_LR.d.size(); ++i)
        H.d[i] = G_LR.d[i] - H_Ab.d[i];
    return H;
}

static FImg applyUSM(const FImg& G_LR, const FImg& H, float k)
{
    FImg out(G_LR.w, G_LR.h);
    for (int i = 0; i < (int)G_LR.d.size(); ++i)
        out.d[i] = std::max(0.f, std::min(255.f, G_LR.d[i] + k * H.d[i]));
    return out;
}

// =============================================================================
// SECTION 3 — CUCKOO SEARCH  [Algorithm 2]
// =============================================================================

namespace cs {

static double logGamma(double x)
{
    static const double c[] = {
         0.99999999999980993,   676.5203681218851,
      -1259.1392167224028,      771.32342877765313,
      -176.61502916214059,       12.507343278686905,
        -0.13857109526572012,     9.9843695780195716e-6,
         1.5056327351493116e-7
    };
    if (x < 0.5)
        return std::log(M_PI / std::sin(M_PI * x)) - logGamma(1.0 - x);
    x -= 1.0;
    double a = c[0];
    for (int i = 1; i <= 8; ++i) a += c[i] / (x + i);
    double t = x + 7.5;
    return 0.5*std::log(2.0*M_PI) + (x+0.5)*std::log(t) - t + std::log(a);
}

static double gammaFn(double x) { return std::exp(logGamma(x)); }

static double computeSigmaM(double beta)
{
    double num = std::sin(M_PI * beta / 2.0) * gammaFn(1.0 + beta);
    double den = gammaFn((1.0 + beta) / 2.0) * beta
                 * std::pow(2.0, (beta - 1.0) / 2.0);
    return std::pow(num / den, 1.0 / beta);
}

static double fitness(const FImg& G_LR, const FImg& H, float k, int step)
{
    double f = 0.0;
    for (int y = 0; y < G_LR.h; y += step)
    for (int x = 0; x < G_LR.w; x += step)
    {
        float val  = std::max(0.f, std::min(255.f,
                         G_LR.at(x,y) + k * H.at(x,y)));
        float valR = std::max(0.f, std::min(255.f,
                         G_LR.sample(x+1,y) + k * H.sample(x+1,y)));
        float valD = std::max(0.f, std::min(255.f,
                         G_LR.sample(x,y+1) + k * H.sample(x,y+1)));
        double gx = valR - val;
        double gy = valD - val;
        f += gx*gx + gy*gy;
    }
    return f;
}

} // namespace cs

static float cuckoosearchOptimiseK(const FImg& G_LR, const FImg& H,
                                   const OLAESplineParams& p)
{
    const int    n    = p.csNests;
    const double beta = p.csBeta;
    const double sigM = cs::computeSigmaM(beta);

    std::cout << "  [OLA diag] CS: nests=" << n << " gen=" << p.csMaxGen
              << " beta=" << beta << " sigmaM=" << sigM
              << " kRange=[" << p.csKMin << "," << p.csKMax << "]"
              << " fitnessStep=" << p.csFitnessStep << std::endl;

    std::mt19937                           rng(p.csSeed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double>       normM(0.0, sigM);
    std::normal_distribution<double>       normN(0.0, 1.0);

    std::vector<double> nests(n);
    for (auto& k : nests)
        k = p.csKMin + uni(rng) * (p.csKMax - p.csKMin);

    std::vector<double> fit(n);
    for (int i = 0; i < n; ++i)
        fit[i] = cs::fitness(G_LR, H, (float)nests[i], p.csFitnessStep);

    int bestIdx = (int)(std::max_element(fit.begin(), fit.end()) - fit.begin());

    std::cout << "  [OLA diag] CS init: best k=" << std::fixed
              << std::setprecision(4) << nests[bestIdx]
              << " fitness=" << fit[bestIdx] << std::endl;

    for (int t = 0; t < p.csMaxGen; ++t)
    {
        int i = std::min((int)(uni(rng) * n), n - 1);
        double mi = normM(rng);
        double ni = normN(rng);
        if (std::abs(ni) < 1e-10) ni = 1e-10;

        double ratio = mi / ni;
        double s = 0.01 * std::pow(std::abs(ratio), 1.0 / beta)
                        * (ratio < 0.0 ? -1.0 : 1.0);
        double levy = std::pow((double)(t + 1), -beta);
        double newK = nests[i] + s * levy;
        newK = std::max(p.csKMin, std::min(p.csKMax, newK));

        double newFit = cs::fitness(G_LR, H, (float)newK, p.csFitnessStep);

        int j = std::min((int)(uni(rng) * n), n - 1);
        if (newFit > fit[j])
        {
            nests[j] = newK;
            fit[j]   = newFit;
            if (newFit > fit[bestIdx]) bestIdx = j;
        }

        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return fit[a] < fit[b]; });
        int abandon = std::max(1, (int)(p.csPa * n));
        for (int w = 0; w < abandon; ++w)
        {
            int wi    = idx[w];
            nests[wi] = p.csKMin + uni(rng) * (p.csKMax - p.csKMin);
            fit[wi]   = cs::fitness(G_LR, H, (float)nests[wi], p.csFitnessStep);
        }
        bestIdx = (int)(std::max_element(fit.begin(), fit.end()) - fit.begin());
    }

    float bestK = (float)nests[bestIdx];
    std::cout << "  [OLA diag] CS result: k=" << std::fixed
              << std::setprecision(4) << bestK
              << " fitness=" << fit[bestIdx] << std::endl;
    return bestK;
}

// =============================================================================
// SECTION 4 — CUBIC B-SPLINE INTERPOLATION  [Eq. 6, 7]
//
// PREFILTER DERIVATION (correct version):
//
// The 1-D B-spline interpolation prefilter converts samples s[k] into
// coefficients c[k] such that:
//   s[k] = sum_j c[j] * beta3(k - j)   for all integer k
//
// This is achieved via the causal/anti-causal IIR with pole z1 = sqrt(3)-2.
//
// For a signal of length N with MIRROR boundary (symmetric extension):
//   The signal is extended as s[-k] = s[k] and s[N-1+k] = s[N-1-k].
//
// Causal initialisation (exact for mirror boundary):
//   c+[0] = sum_{k=0}^{inf} (s[k] + s[-k]) * z1^k
//           but s[-k] = s[k] for mirror, so:
//         = s[0] + 2 * sum_{k=1}^{N-1} s[k] * z1^k   (truncated, |z1|^30 < 1e-6)
//
// This is the formula used by scipy.ndimage and OpenCV.
// =============================================================================

static float beta3(float t)
{
    float a = std::abs(t);
    if (a < 1.f)
        return 2.f/3.f - 0.5f * a * a * (2.f - a);
    if (a < 2.f) {
        float d = 2.f - a;
        return d * d * d / 6.f;
    }
    return 0.f;
}

static void bsplinePrefilter1D(std::vector<float>& c, int n)
{
    if (n < 2) return;

    const float z1     = std::sqrt(3.f) - 2.f;   // ≈ -0.2679
    const float lambda = (1.f - z1) * (1.f - 1.f / z1);
    // lambda = (1 - z1)(1 - 1/z1) = (1 - z1)(z1 - 1)/z1 = -(1-z1)^2 / z1
    // With z1 = sqrt(3)-2: lambda ≈ 6.0  [gain correction factor]

    // Apply gain
    for (auto& v : c) v *= lambda;

    // -----------------------------------------------------------------------
    // Causal initialisation — mirror (symmetric) boundary:
    //
    //   c+[0] = c[0] + 2 * sum_{k=1}^{N-1} c[k] * z1^k
    //
    // Since |z1| ≈ 0.268, z1^k decays to < 1e-7 by k=30.
    // We use min(N-1, 30) terms.
    // -----------------------------------------------------------------------
    {
        int horizon = std::min(n - 1, 30);
        float zk  = z1;
        float sum = c[0]; // k=0 term
        for (int k = 1; k <= horizon; ++k, zk *= z1)
            sum += 2.f * c[k] * zk;
        c[0] = sum;
    }

    // Causal pass: c+[k] = c[k] + z1 * c+[k-1]
    for (int i = 1; i < n; ++i)
        c[i] += z1 * c[i - 1];

    // -----------------------------------------------------------------------
    // Anti-causal initialisation — mirror boundary:
    //
    //   c-[N-1] = (z1 / (z1^2 - 1)) * (c+[N-1] + z1 * c+[N-2])
    //
    // This is the standard formula from Thevenaz 2000, correct for both
    // mirror and anti-mirror boundaries.
    // -----------------------------------------------------------------------
    c[n-1] = (z1 / (z1*z1 - 1.f)) * (z1 * c[n-2] + c[n-1]);

    // Anti-causal pass: c-[k] = z1 * (c-[k+1] - c+[k])
    for (int i = n - 2; i >= 0; --i)
        c[i] = z1 * (c[i+1] - c[i]);
}

static FImg computeBSplineCoeffs(const FImg& src)
{
    FImg C = src;

    std::vector<float> row(src.w);
    for (int y = 0; y < src.h; ++y)
    {
        for (int x = 0; x < src.w; ++x) row[x] = C.at(x, y);
        bsplinePrefilter1D(row, src.w);
        for (int x = 0; x < src.w; ++x) C.at(x, y) = row[x];
    }

    std::vector<float> col(src.h);
    for (int x = 0; x < src.w; ++x)
    {
        for (int y = 0; y < src.h; ++y) col[y] = C.at(x, y);
        bsplinePrefilter1D(col, src.h);
        for (int y = 0; y < src.h; ++y) C.at(x, y) = col[y];
    }
    return C;
}

// Verify prefilter correctness: reconstruct at integer positions and check
// that we recover the original values (max error should be < 0.1).
static void verifyBSplinePrefilter(const FImg& original, const FImg& coeffs)
{
    float maxErr = 0.f, sumErr = 0.f;
    int count = 0;
    // Check a sample of pixels
    int step = std::max(1, std::min(original.w, original.h) / 10);
    for (int y = 0; y < original.h; y += step)
    for (int x = 0; x < original.w; x += step)
    {
        // At integer position, evalBSpline should reconstruct original value
        float fx = (float)x, fy = (float)y;
        int x0 = x, y0 = y;
        float acc = 0.f;
        for (int j = y0 - 1; j <= y0 + 2; ++j)
        for (int i = x0 - 1; i <= x0 + 2; ++i)
            acc += coeffs.sample(i,j) * beta3(fx-(float)i) * beta3(fy-(float)j);
        float err = std::abs(acc - original.at(x, y));
        if (err > maxErr) maxErr = err;
        sumErr += err;
        ++count;
    }
    std::cout << "  [OLA diag] B-spline prefilter verify: maxErr="
              << std::fixed << std::setprecision(4) << maxErr
              << " meanErr=" << (count > 0 ? sumErr/count : 0.f)
              << " (should be < 0.5)" << std::endl;
}

static float evalBSpline(const FImg& C, float fx, float fy)
{
    int x0 = (int)std::floor(fx);
    int y0 = (int)std::floor(fy);
    float acc = 0.f;
    for (int j = y0 - 1; j <= y0 + 2; ++j)
    for (int i = x0 - 1; i <= x0 + 2; ++i)
        acc += C.sample(i, j) * beta3(fx - (float)i) * beta3(fy - (float)j);
    return acc;
}

static FImg bsplineUpscale(const FImg& G_SLR, int scale)
{
    FImg C = computeBSplineCoeffs(G_SLR);

    // Verify the prefilter is working correctly
    verifyBSplinePrefilter(G_SLR, C);

    // Print coefficient stats to check for blow-up
    printImgStats("B-spline coefficients", C);

    int outW = G_SLR.w * scale;
    int outH = G_SLR.h * scale;
    FImg G_HR(outW, outH);

    const float fscale = (float)scale;
    for (int oy = 0; oy < outH; ++oy)
    for (int ox = 0; ox < outW;  ++ox)
    {
        // Centre-aligned coordinate mapping
        float fx = (ox + 0.5f) / fscale - 0.5f;
        float fy = (oy + 0.5f) / fscale - 0.5f;
        float v  = evalBSpline(C, fx, fy);
        G_HR.at(ox, oy) = std::max(0.f, std::min(255.f, v));
    }
    return G_HR;
}

// =============================================================================
// SECTION 5 — CANNY EDGE DETECTION
// =============================================================================

static FImg cannyEdgeDetect(const FImg& src, float lowT, float highT)
{
    int W = src.w, H = src.h;

    static const float gk[5][5] = {
        {2,  4,  5,  4, 2},
        {4,  9, 12,  9, 4},
        {5, 12, 15, 12, 5},
        {4,  9, 12,  9, 4},
        {2,  4,  5,  4, 2}
    };
    const float gkSum = 159.f;

    FImg sm(W, H);
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
    {
        float acc = 0.f;
        for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx)
            acc += src.sample(x+dx, y+dy) * gk[dy+2][dx+2];
        sm.at(x, y) = acc / gkSum;
    }

    FImg mag(W, H), ang(W, H);
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
    {
        float gx = -sm.sample(x-1,y-1) + sm.sample(x+1,y-1)
                   -2.f*sm.sample(x-1,y) + 2.f*sm.sample(x+1,y)
                   -sm.sample(x-1,y+1) + sm.sample(x+1,y+1);
        float gy = -sm.sample(x-1,y-1) - 2.f*sm.sample(x,y-1)
                   -sm.sample(x+1,y-1) + sm.sample(x-1,y+1)
                   +2.f*sm.sample(x,y+1) + sm.sample(x+1,y+1);
        mag.at(x,y) = std::sqrt(gx*gx + gy*gy);
        ang.at(x,y) = std::atan2(gy, gx);
    }

    FImg nms(W, H);
    for (int y = 1; y < H-1; ++y)
    for (int x = 1; x < W-1; ++x)
    {
        float m   = mag.at(x, y);
        float deg = ang.at(x, y) * 180.f / (float)M_PI;
        if (deg < 0.f) deg += 180.f;

        float m1, m2;
        if      (deg < 22.5f || deg >= 157.5f) { m1=mag.at(x-1,y);   m2=mag.at(x+1,y);   }
        else if (deg < 67.5f)                  { m1=mag.at(x-1,y+1); m2=mag.at(x+1,y-1); }
        else if (deg < 112.5f)                 { m1=mag.at(x,y-1);   m2=mag.at(x,y+1);   }
        else                                   { m1=mag.at(x-1,y-1); m2=mag.at(x+1,y+1); }

        nms.at(x,y) = (m >= m1 && m >= m2) ? m : 0.f;
    }

    FImg edges(W, H);
    for (int i = 0; i < W*H; ++i)
    {
        float v = nms.d[i];
        edges.d[i] = (v >= highT) ? 255.f : (v >= lowT) ? 128.f : 0.f;
    }
    for (int y = 1; y < H-1; ++y)
    for (int x = 1; x < W-1; ++x)
    {
        if (edges.at(x,y) != 128.f) continue;
        bool conn = false;
        for (int dy = -1; dy <= 1 && !conn; ++dy)
        for (int dx = -1; dx <= 1 && !conn; ++dx)
            conn = (edges.at(x+dx, y+dy) == 255.f);
        edges.at(x,y) = conn ? 255.f : 0.f;
    }

    // Count edge pixels
    int edgeCount = 0;
    for (float v : edges.d) if (v > 128.f) edgeCount++;
    std::cout << "  [OLA diag] Canny: edgePixels=" << edgeCount
              << " of " << W*H << " total" << std::endl;

    return edges;
}

// =============================================================================
// SECTION 6 — e-SPLINE EDGE EXPANSION  [Eq. 9-15]
// =============================================================================

static FImg eSplineEdgeExpansion(const FImg& G_HR, const FImg& edgeMap)
{
    int W = G_HR.w, H = G_HR.h;
    FImg G_e(W, H);

    int modifiedCount = 0;

    for (int y = 2; y < H - 2; ++y)
    for (int x = 2; x < W - 2; ++x)
    {
        if (edgeMap.at(x, y) < 128.f) continue;

        float SDh =
            0.25f * (G_HR.sample(x-1,y-1) - G_HR.sample(x-1,y+1)) +
            0.50f * (G_HR.sample(x,  y-1) - G_HR.sample(x,  y+1)) +
            0.25f * (G_HR.sample(x+1,y-1) - G_HR.sample(x+1,y+1));

        float SDv =
            0.25f * (G_HR.sample(x-1,y-1) - G_HR.sample(x+1,y-1)) +
            0.50f * (G_HR.sample(x-1,y  ) - G_HR.sample(x+1,y  )) +
            0.25f * (G_HR.sample(x-1,y+1) - G_HR.sample(x+1,y+1));

        if (std::abs(SDh) >= std::abs(SDv))
        {
            float newYm1 = 0.5f*(G_HR.sample(x,y-1)+G_HR.sample(x,y-2));
            float newYp1 = 0.5f*(G_HR.sample(x,y+1)+G_HR.sample(x,y+2));
            G_e.at(x, y-1) = newYm1 - G_HR.at(x, y-1);
            G_e.at(x, y+1) = newYp1 - G_HR.at(x, y+1);
        }
        else
        {
            float newXp1 = 0.5f*(G_HR.sample(x+1,y)+G_HR.sample(x+2,y));
            float newXm1 = 0.5f*(G_HR.sample(x-1,y)+G_HR.sample(x-2,y));
            G_e.at(x+1,y) = newXp1 - G_HR.at(x+1,y);
            G_e.at(x-1,y) = newXm1 - G_HR.at(x-1,y);
        }
        ++modifiedCount;
    }

    std::cout << "  [OLA diag] Edge expansion: modified around "
              << modifiedCount << " edge pixels" << std::endl;

    FImg G_RHR(W, H);
    for (int i = 0; i < W*H; ++i)
        G_RHR.d[i] = std::max(0.f, std::min(255.f, G_HR.d[i] + G_e.d[i]));
    return G_RHR;
}

// =============================================================================
// SECTION 7 — SINGLE-CHANNEL PIPELINE
// =============================================================================

static FImg olaESplineSingleChannel(const FImg& G_LR,
                                     const OLAESplineParams& p)
{
    std::cout << "  [OLA diag] --- Channel pipeline start ---" << std::endl;
    printImgStats("G_LR input", G_LR);

    // Stage 1
    FImg H_Ab = laAdaptiveGaussianBlur(G_LR);
    printImgStats("H_Ab (adaptive blur)", H_Ab);

    // Stage 2
    FImg H = computeHPF(G_LR, H_Ab);
    printImgStats("H (HPF)", H);

    // Stage 3
    float k = cuckoosearchOptimiseK(G_LR, H, p);

    // Stage 4
    FImg G_SLR = applyUSM(G_LR, H, k);
    printImgStats("G_SLR (sharpened LR)", G_SLR);

    // Stage 5
    FImg G_HR = bsplineUpscale(G_SLR, p.scaleFactor);
    printImgStats("G_HR (B-spline upscaled)", G_HR);

    // Stage 6
    FImg edgeMap = cannyEdgeDetect(G_HR, p.cannyLow, p.cannyHigh);

    // Stage 7
    FImg G_RHR = eSplineEdgeExpansion(G_HR, edgeMap);
    printImgStats("G_RHR (final output)", G_RHR);

    std::cout << "  [OLA diag] --- Channel pipeline end ---" << std::endl;
    return G_RHR;
}

// =============================================================================
// SECTION 8 — PUBLIC RGBA ENTRY POINT
// =============================================================================

void scaleOLAESpline(const unsigned char* input,  int inW,  int inH,
                           unsigned char* output, int outW, int outH,
                     const OLAESplineParams& params)
{
    (void)outW; (void)outH;
    const int oW = inW * params.scaleFactor;
    const int oH = inH * params.scaleFactor;

    const char* chName[] = {"R","G","B"};
    for (int ch = 0; ch < 3; ++ch)
    {
        std::cout << "  [OLA] Processing channel " << chName[ch] << std::endl;

        FImg G_LR(inW, inH);
        for (int y = 0; y < inH; ++y)
        for (int x = 0; x < inW;  ++x)
            G_LR.at(x,y) = (float)input[(y*inW+x)*4+ch];

        FImg G_RHR = olaESplineSingleChannel(G_LR, params);

        for (int y = 0; y < oH; ++y)
        for (int x = 0; x < oW; ++x)
        {
            float v = G_RHR.at(x,y);
            output[(y*oW+x)*4+ch] =
                (unsigned char)(std::max(0.f,std::min(255.f,v))+0.5f);
        }
    }

    // Alpha: nearest-neighbour
    for (int oy = 0; oy < oH; ++oy)
    for (int ox = 0; ox < oW; ++ox)
    {
        int sx = std::min(ox / params.scaleFactor, inW-1);
        int sy = std::min(oy / params.scaleFactor, inH-1);
        output[(oy*oW+ox)*4+3] = input[(sy*inW+sx)*4+3];
    }
}
