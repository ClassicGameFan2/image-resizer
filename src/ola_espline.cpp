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

    // Clamp-to-edge out-of-bounds sampling
    float sample(int x, int y) const
    {
        x = std::max(0, std::min(x, w - 1));
        y = std::max(0, std::min(y, h - 1));
        return d[y * w + x];
    }
};

// =============================================================================
// SECTION 1 — LOCAL ADAPTIVE (LA) GAUSSIAN FILTERING  [Algorithm 1]
//
// From Algorithm 1 image:
//   1. Find V_max, V_min = global max/min of all local 3x3 variances.
//   2. S = (V_max - V_min) / 6
//   3. For each pixel:
//        Compute local 3x3 mean m and variance v.
//        C_p via step function:
//          v > V_max - S          → C_p = 2   (high variance, edge: more blur)
//          v > V_max - 2S         → C_p = 3
//          v > V_max - 3S         → C_p = 4
//          v > V_max - 4S         → C_p = 8
//          v > V_max - 5S         → C_p = 16
//          else                   → C_p = 32  (low variance, smooth: less blur)
//        Kernel: g_f = 1/(C_p+12) * |1  2  1|
//                                    |1  Cp 1|
//                                    |1  2  1|
//        H_Ab(x,y) = sum_{i,j=-1}^{1} g_f(i,j) * G_LR(x+i, y+j)
//
// The normalization denominator is (C_p + 12) as explicitly stated in
// Algorithm 1, even though the kernel element sum is (C_p + 10).
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
    if (S < 1e-9f) return 32; // uniform image — no edges
    if      (v > V_max - 1.f * S) return 2;
    else if (v > V_max - 2.f * S) return 3;
    else if (v > V_max - 3.f * S) return 4;
    else if (v > V_max - 4.f * S) return 8;
    else if (v > V_max - 5.f * S) return 16;
    else                           return 32;
}

// Kernel surrounding weights (non-centre positions), from Algorithm 1 image:
//   row dy=-1:  1  2  1
//   row dy= 0:  1  *  1   (* = C_p, handled separately)
//   row dy=+1:  1  2  1
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

    FImg dst(src.w, src.h);

    for (int y = 0; y < src.h; ++y)
    for (int x = 0; x < src.w; ++x)
    {
        float m, v;
        localStats(src, x, y, m, v);

        int   Cp   = selectCp(v, V_max, S);
        float norm = static_cast<float>(Cp) + 12.f; // Algorithm 1 denominator

        // Centre contribution
        float acc = static_cast<float>(Cp) * src.at(x, y);
        // Surrounding contributions
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            acc += kW[dy + 1][dx + 1] * src.sample(x + dx, y + dy);
        }
        dst.at(x, y) = acc / norm;
    }
    return dst;
}

// =============================================================================
// SECTION 2 — HPF EXTRACTION AND USM SHARPENING  [Eq. 4, 1]
//
// Eq.(4): H(x,y)  = G_LR(x,y) - H_Ab(x,y)
// Eq.(1): G_SLR   = G_LR + k * H(x,y)
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
// SECTION 3 — CUCKOO SEARCH OPTIMISATION  [Algorithm 2, Section 4.2, 4.3]
//
// Algorithm 2:
//   Result: Suitable value of k
//   Objective: f(x), x=(x1,...,x_d)
//   Input: HPF image H(x,y) and LR image G_LR
//   Generate initial population of n host nests.
//   while t < g:
//     Get cuckoo randomly (say i), perform Levy flight to get new solution.
//     Evaluate fitness f_i proportional to f(x_i).
//     Choose nest j randomly.
//     if f_i > f_j: replace j with new solution
//     else:         keep j
//
// Levy flight [Eq.5, Section 4.3]:
//   c(t+1) = c(t) + s * l(beta)
//   l(beta) = t^{-beta},  1 < beta < 3
//   s = 0.01 * (m_i/n_i)^{1/beta}
//   m_i ~ N(0, sigma_m^2),  n_i ~ N(0, 1)   [sigma_n = 1]
//   sigma_m = [sin(pi*beta/2) * Gamma(1+beta)] /
//             [Gamma((1+beta)/2) * beta * 2^{(beta-1)/2}]
//
// Fitness function: sum of squared image gradients of G_SLR = G_LR + k*H.
// Maximising gradient energy encourages edge/detail restoration, consistent
// with the paper's pre-upscaling sharpening intent.
// csFitnessStep subsamples pixels to control speed.
// =============================================================================

namespace cs {

static double logGamma(double x)
{
    // Lanczos approximation, g=7
    static const double c[] = {
         0.99999999999980993,
       676.5203681218851,
      -1259.1392167224028,
       771.32342877765313,
      -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
         9.9843695780195716e-6,
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

// sigma_m from Mantegna's algorithm [Section 4.3]:
// sigma_m = ( sin(pi*beta/2) * Gamma(1+beta) ) /
//           ( Gamma((1+beta)/2) * beta * 2^{(beta-1)/2} )
// then raised to power 1/beta (standard Mantegna normalisation).
static double computeSigmaM(double beta)
{
    double num = std::sin(M_PI * beta / 2.0) * gammaFn(1.0 + beta);
    double den = gammaFn((1.0 + beta) / 2.0) * beta
                 * std::pow(2.0, (beta - 1.0) / 2.0);
    return std::pow(num / den, 1.0 / beta);
}

// Fitness: sum of squared finite-difference gradients of G_SLR.
// step=1 evaluates every pixel; step=N evaluates every N-th pixel.
static double fitness(const FImg& G_LR, const FImg& H, float k, int step)
{
    double f = 0.0;
    for (int y = 0; y < G_LR.h; y += step)
    for (int x = 0; x < G_LR.w; x += step)
    {
        float val  = std::max(0.f, std::min(255.f,
                         G_LR.at(x, y) + k * H.at(x, y)));
        float valR = std::max(0.f, std::min(255.f,
                         G_LR.sample(x+1, y) + k * H.sample(x+1, y)));
        float valD = std::max(0.f, std::min(255.f,
                         G_LR.sample(x, y+1) + k * H.sample(x, y+1)));
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

    std::mt19937                           rng(p.csSeed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    // m_i ~ N(0, sigma_m^2): std dev = sigM
    std::normal_distribution<double>       normM(0.0, sigM);
    // n_i ~ N(0, 1): sigma_n = 1 [paper]
    std::normal_distribution<double>       normN(0.0, 1.0);

    // Initialise nests randomly in [kMin, kMax]
    std::vector<double> nests(n);
    for (auto& k : nests)
        k = p.csKMin + uni(rng) * (p.csKMax - p.csKMin);

    std::vector<double> fit(n);
    for (int i = 0; i < n; ++i)
        fit[i] = cs::fitness(G_LR, H, (float)nests[i], p.csFitnessStep);

    int bestIdx = (int)(std::max_element(fit.begin(), fit.end()) - fit.begin());

    // Main CS loop: while t < g  [Algorithm 2]
    for (int t = 0; t < p.csMaxGen; ++t)
    {
        // Pick cuckoo i randomly
        int i = std::min((int)(uni(rng) * n), n - 1);

        // Levy flight [Eq.5 + Mantegna step-size formulas]
        double mi = normM(rng);                     // m_i ~ N(0, sigma_m^2)
        double ni = normN(rng);                     // n_i ~ N(0, 1)
        if (std::abs(ni) < 1e-10) ni = 1e-10;

        // s = 0.01 * |m_i/n_i|^{1/beta} * sign(m_i/n_i)
        double ratio = mi / ni;
        double s = 0.01 * std::pow(std::abs(ratio), 1.0 / beta)
                        * (ratio < 0.0 ? -1.0 : 1.0);

        // l(beta) = t^{-beta}  [use t+1 to avoid 0^{-beta} at t=0]
        double levy = std::pow((double)(t + 1), -beta);

        // c(t+1) = c(t) + s * l(beta)  [Eq.5]
        double newK = nests[i] + s * levy;
        newK = std::max(p.csKMin, std::min(p.csKMax, newK));

        double newFit = cs::fitness(G_LR, H, (float)newK, p.csFitnessStep);

        // Choose random host nest j
        int j = std::min((int)(uni(rng) * n), n - 1);

        // Algorithm 2: if f_i > f_j replace j, else keep j
        if (newFit > fit[j])
        {
            nests[j] = newK;
            fit[j]   = newFit;
            if (newFit > fit[bestIdx]) bestIdx = j;
        }

        // Abandon worst pa fraction of nests [Section 4.2]
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

        // Update global best
        bestIdx = (int)(std::max_element(fit.begin(), fit.end()) - fit.begin());
    }

    return (float)nests[bestIdx];
}

// =============================================================================
// SECTION 4 — CUBIC B-SPLINE INTERPOLATION  [Eq. 6, 7]
//
// Eq.(7): beta^3(x):
//   2/3 - (1/2)|x|^2 * (2-|x|)    for 0 <= |x| < 1
//   (1/6)(2-|x|)^3                  for 1 <= |x| < 2
//   0                                otherwise
//
// Eq.(6): G_HR(x,y) = sum_i sum_j C(i,j) * beta^3(x-i) * beta^3(y-j)
//
// C(i,j) are computed from G_SLR via the B-spline interpolation prefilter
// (causal + anti-causal IIR passes) with pole z1 = sqrt(3) - 2 ≈ -0.2679.
//
// COORDINATE MAPPING (centre-aligned, consistent with other scalers):
//   Input pixel (ix, iy) centre is at coordinate (ix, iy).
//   Output pixel (ox, oy) maps to input coordinate:
//     fx = (ox + 0.5) / scale - 0.5
//     fy = (oy + 0.5) / scale - 0.5
//   This matches the convention used by scaleBicubic/scaleLanczos3/FSR.
// =============================================================================

// Cubic B-spline basis function [Eq.7]
static float beta3(float t)
{
    float a = std::abs(t);
    if (a < 1.f)
        return 2.f/3.f - 0.5f * a * a * (2.f - a);
    if (a < 2.f)
    {
        float d = 2.f - a;
        return d * d * d / 6.f;
    }
    return 0.f;
}

// 1-D B-spline interpolation prefilter.
// Converts image samples to B-spline coefficients C such that the
// reconstructed signal exactly interpolates the input at integer positions.
// Pole: z1 = sqrt(3) - 2 ≈ -0.2679.
// Reference: Thévenaz, Blu, Unser, IEEE Trans. Med. Imag. 2000.
//
// FIX vs previous version: the causal initialisation no longer divides by z1
// (which caused divergence). Instead we use a truncated geometric series
// which converges rapidly because |z1|^k → 0 for k → inf.
static void bsplinePrefilter1D(std::vector<float>& c, int n)
{
    if (n < 2) return;

    const float z1     = std::sqrt(3.f) - 2.f;   // ≈ -0.2679
    const float lambda = (1.f - z1) * (1.f - 1.f / z1);
    // lambda ≈ 6.0  — this is the gain that the IIR filter must undo.

    // Apply gain correction
    for (auto& v : c) v *= lambda;

    // -----------------------------------------------------------------------
    // Causal initialisation (mirror boundary):
    //   c+[0] = sum_{k=0}^{N-1} c[k] * z1^k  +  z1^N * sum_{k=0}^{N-1} c[N-1-k] * z1^k
    //         = sum_{k=0}^{N-1} c[k] * z1^k  +  z1^{2N-1} * sum_{k=0}^{N-1} c[k] * z1^{-k}
    //
    // Because |z1| < 0.3, z1^k decays to machine precision in ~30 terms.
    // We accumulate the series directly — NO division by z1 (fixes divergence).
    // -----------------------------------------------------------------------
    {
        // How many terms to include: |z1|^horizon < 1e-7
        // |(-0.2679)|^horizon < 1e-7  → horizon > 7/log10(1/0.2679) ≈ 12
        // Use min(n, 32) to be safe.
        int horizon = std::min(n, 32);

        // Forward sum: sum c[k] * z1^k
        float zk   = 1.f;
        float sum  = 0.f;
        for (int k = 0; k < horizon; ++k)
        {
            sum += c[k] * zk;
            zk  *= z1;
        }

        // Backward sum: z1^{2N-1} * sum c[N-1-k] * z1^{-k}
        // = z1^{N} * sum c[N-1-k] * z1^{N-1-k}   (reversed forward sum)
        float zN  = std::pow(z1, (float)n);
        float bkz = 1.f;
        float bsum = 0.f;
        for (int k = 0; k < horizon; ++k)
        {
            bsum += c[n - 1 - k] * bkz;   // bkz = z1^k
            bkz  *= z1;
        }
        // Mirror contribution: z1^N * bsum (reversed)
        // Standard formula: c+[0] = fwd_sum / (1 - z1^{2N})
        // For large N, z1^{2N} ≈ 0, so denominator ≈ 1.
        float denom = 1.f - std::pow(z1, (float)(2 * n));
        // Full mirror initialisation:
        c[0] = (sum + zN * bsum) / denom;
    }

    // Causal pass
    for (int i = 1; i < n; ++i)
        c[i] += z1 * c[i - 1];

    // Anti-causal initialisation
    c[n-1] = (z1 / (z1*z1 - 1.f)) * (z1 * c[n-2] + c[n-1]);

    // Anti-causal pass
    for (int i = n - 2; i >= 0; --i)
        c[i] = z1 * (c[i+1] - c[i]);
}

// 2-D separable B-spline coefficient computation.
static FImg computeBSplineCoeffs(const FImg& src)
{
    FImg C = src; // copy

    // Row-wise prefilter
    std::vector<float> row(src.w);
    for (int y = 0; y < src.h; ++y)
    {
        for (int x = 0; x < src.w; ++x) row[x] = C.at(x, y);
        bsplinePrefilter1D(row, src.w);
        for (int x = 0; x < src.w; ++x) C.at(x, y) = row[x];
    }

    // Column-wise prefilter
    std::vector<float> col(src.h);
    for (int x = 0; x < src.w; ++x)
    {
        for (int y = 0; y < src.h; ++y) col[y] = C.at(x, y);
        bsplinePrefilter1D(col, src.h);
        for (int y = 0; y < src.h; ++y) C.at(x, y) = col[y];
    }
    return C;
}

// Evaluate B-spline surface at continuous (fx, fy) — Eq.(6).
// Support: 4×4 neighbourhood (beta^3 is non-zero for |t| < 2).
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

// Upscale by integer factor using centre-aligned coordinate mapping.
// FIX: fx = (ox + 0.5) / scale - 0.5  (was: fx = ox / scale).
static FImg bsplineUpscale(const FImg& G_SLR, int scale)
{
    FImg C   = computeBSplineCoeffs(G_SLR);
    int  outW = G_SLR.w * scale;
    int  outH = G_SLR.h * scale;
    FImg G_HR(outW, outH);

    const float fscale = (float)scale;
    for (int oy = 0; oy < outH; ++oy)
    for (int ox = 0; ox < outW;  ++ox)
    {
        // Centre-aligned: maps output pixel centre to input coordinate.
        float fx = (ox + 0.5f) / fscale - 0.5f;
        float fy = (oy + 0.5f) / fscale - 0.5f;
        float v  = evalBSpline(C, fx, fy);
        G_HR.at(ox, oy) = std::max(0.f, std::min(255.f, v));
    }
    return G_HR;
}

// =============================================================================
// SECTION 5 — CANNY EDGE DETECTION
//
// Standard four-stage Canny pipeline:
//   1. 5×5 Gaussian smoothing (σ ≈ 1.0)
//   2. Sobel gradient magnitude and direction
//   3. Non-maximum suppression (1-pixel-wide edges)
//   4. Double threshold (strong/weak) + hysteresis edge linking
//
// Returns: 255 = edge, 0 = non-edge.
// =============================================================================

static FImg cannyEdgeDetect(const FImg& src, float lowT, float highT)
{
    int W = src.w, H = src.h;

    // 1. Gaussian smoothing
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

    // 2. Sobel gradients
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
        mag.at(x, y) = std::sqrt(gx*gx + gy*gy);
        ang.at(x, y) = std::atan2(gy, gx);
    }

    // 3. Non-maximum suppression
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

        nms.at(x, y) = (m >= m1 && m >= m2) ? m : 0.f;
    }

    // 4. Double threshold + hysteresis
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
        edges.at(x, y) = conn ? 255.f : 0.f;
    }
    return edges;
}

// =============================================================================
// SECTION 6 — e-SPLINE EDGE EXPANSION  [Section 4.5, Eqs. 9-15]
//
// For each detected edge pixel at (x,y) in G_HR:
//
//   SDh [Eq.9] — horizontal change (left-right), detects vertical edges:
//     SDh = 1/4*[E(x-1,y-1)-E(x-1,y+1)]
//           + 1/2*[E(x,y-1)-E(x,y+1)]
//           + 1/4*[E(x+1,y-1)-E(x+1,y+1)]
//
//   SDv [Eq.10] — vertical change (up-down), detects horizontal edges:
//     SDv = 1/4*[E(x-1,y-1)-E(x+1,y-1)]
//           + 1/2*[E(x-1,y)-E(x+1,y)]
//           + 1/4*[E(x-1,y+1)-E(x+1,y+1)]
//
//   (In our FImg: x=column, y=row. Paper Table 1: x=row, y=column.
//    The formulas are applied faithfully in code coordinates.)
//
//   if |SDh| >= |SDv|  → vertical edge → expand left/right neighbours:
//     E(x, y-1) = 1/2*[E(x,y-1) + E(x,y-2)]   [Eq.11]
//     E(x, y+1) = 1/2*[E(x,y+1) + E(x,y+2)]   [Eq.12]
//
//   if |SDh| <= |SDv|  → horizontal edge → expand above/below neighbours:
//     E(x+1, y) = 1/2*[E(x+1,y) + E(x+2,y)]   [Eq.13]
//     E(x-1, y) = 1/2*[E(x-1,y) + E(x-2,y)]   [Eq.14]
//
// G_e stores the DELTA only (correction = new_value - original_value).
// All reads from G_HR (original, unmodified).
// G_RHR = G_HR + G_e  [Eq.15]  — Figure 2 final '+' node.
// =============================================================================

static FImg eSplineEdgeExpansion(const FImg& G_HR, const FImg& edgeMap)
{
    int W = G_HR.w, H = G_HR.h;
    FImg G_e(W, H); // zero-initialised delta image

    for (int y = 2; y < H - 2; ++y)
    for (int x = 2; x < W - 2; ++x)
    {
        if (edgeMap.at(x, y) < 128.f) continue;

        // SDh [Eq.9]: measures left-right change across columns
        float SDh =
            0.25f * (G_HR.sample(x-1, y-1) - G_HR.sample(x-1, y+1)) +
            0.50f * (G_HR.sample(x,   y-1) - G_HR.sample(x,   y+1)) +
            0.25f * (G_HR.sample(x+1, y-1) - G_HR.sample(x+1, y+1));

        // SDv [Eq.10]: measures up-down change across rows
        float SDv =
            0.25f * (G_HR.sample(x-1, y-1) - G_HR.sample(x+1, y-1)) +
            0.50f * (G_HR.sample(x-1, y  ) - G_HR.sample(x+1, y  )) +
            0.25f * (G_HR.sample(x-1, y+1) - G_HR.sample(x+1, y+1));

        if (std::abs(SDh) >= std::abs(SDv))
        {
            // Vertical edge: blend left/right neighbours with their outer peers
            float newYm1 = 0.5f * (G_HR.sample(x, y-1) + G_HR.sample(x, y-2));
            float newYp1 = 0.5f * (G_HR.sample(x, y+1) + G_HR.sample(x, y+2));
            G_e.at(x, y-1) = newYm1 - G_HR.at(x, y-1);
            G_e.at(x, y+1) = newYp1 - G_HR.at(x, y+1);
        }
        else
        {
            // Horizontal edge: blend above/below neighbours with their outer peers
            float newXp1 = 0.5f * (G_HR.sample(x+1, y) + G_HR.sample(x+2, y));
            float newXm1 = 0.5f * (G_HR.sample(x-1, y) + G_HR.sample(x-2, y));
            G_e.at(x+1, y) = newXp1 - G_HR.at(x+1, y);
            G_e.at(x-1, y) = newXm1 - G_HR.at(x-1, y);
        }
    }

    // Eq.(15): G_RHR = G_HR + G_e
    FImg G_RHR(W, H);
    for (int i = 0; i < W * H; ++i)
        G_RHR.d[i] = std::max(0.f, std::min(255.f, G_HR.d[i] + G_e.d[i]));

    return G_RHR;
}

// =============================================================================
// SECTION 7 — SINGLE-CHANNEL OLA e-SPLINE PIPELINE
// =============================================================================

static FImg olaESplineSingleChannel(const FImg& G_LR,
                                     const OLAESplineParams& p)
{
    // Stage 1: LA Gaussian blur → H_Ab  [Algorithm 1]
    FImg H_Ab = laAdaptiveGaussianBlur(G_LR);

    // Stage 2: HPF extraction  [Eq.4]
    FImg H = computeHPF(G_LR, H_Ab);

    // Stage 3: CS optimisation → k  [Algorithm 2]
    float k = cuckoosearchOptimiseK(G_LR, H, p);

    // Stage 4: Sharpened LR  [Eq.1]
    FImg G_SLR = applyUSM(G_LR, H, k);

    // Stage 5: B-spline upscale → G_HR  [Eq.6]
    FImg G_HR = bsplineUpscale(G_SLR, p.scaleFactor);

    // Stage 6: Canny edge detection
    FImg edgeMap = cannyEdgeDetect(G_HR, p.cannyLow, p.cannyHigh);

    // Stage 7: e-spline edge expansion → G_RHR = G_HR + G_e  [Eq.15]
    FImg G_RHR = eSplineEdgeExpansion(G_HR, edgeMap);

    return G_RHR;
}

// =============================================================================
// SECTION 8 — PUBLIC RGBA ENTRY POINT
// =============================================================================

void scaleOLAESpline(const unsigned char* input,  int inW,  int inH,
                           unsigned char* output, int outW, int outH,
                     const OLAESplineParams& params)
{
    (void)outW;
    (void)outH;

    const int oW = inW * params.scaleFactor;
    const int oH = inH * params.scaleFactor;

    // Process R, G, B channels independently
    for (int ch = 0; ch < 3; ++ch)
    {
        // Extract channel [0, 255]
        FImg G_LR(inW, inH);
        for (int y = 0; y < inH; ++y)
        for (int x = 0; x < inW; ++x)
            G_LR.at(x, y) = (float)input[(y * inW + x) * 4 + ch];

        // Run full OLA e-spline pipeline
        FImg G_RHR = olaESplineSingleChannel(G_LR, params);

        // Write back clamped to [0,255]
        for (int y = 0; y < oH; ++y)
        for (int x = 0; x < oW; ++x)
        {
            float v = G_RHR.at(x, y);
            output[(y * oW + x) * 4 + ch] =
                (unsigned char)(std::max(0.f, std::min(255.f, v)) + 0.5f);
        }
    }

    // Alpha: nearest-neighbour (preserves transparency exactly)
    for (int oy = 0; oy < oH; ++oy)
    for (int ox = 0; ox < oW; ++ox)
    {
        int sx = std::min(ox / params.scaleFactor, inW - 1);
        int sy = std::min(oy / params.scaleFactor, inH - 1);
        output[(oy * oW + ox) * 4 + 3] = input[(sy * inW + sx) * 4 + 3];
    }
}
