// =============================================================================
// ola_espline.cpp — final solution
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

    float mean() const
    {
        if (d.empty()) return 0.f;
        double s = 0.0;
        for (float v : d) s += v;
        return (float)(s / d.size());
    }

    void stats(float& mn, float& mx, float& mean_) const
    {
        mn = mx = d.empty() ? 0.f : d[0];
        double s = 0.0;
        for (float v : d) {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            s += v;
        }
        mean_ = d.empty() ? 0.f : (float)(s / d.size());
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
              << "  size=" << img.w << "x" << img.h << std::endl;
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

static void globalVarianceBounds(const FImg& src, float& V_max, float& V_min)
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

    int cpCount[6] = {};
    static const int cpVals[6] = {2,3,4,8,16,32};

    FImg dst(src.w, src.h);
    for (int y = 0; y < src.h; ++y)
    for (int x = 0; x < src.w; ++x)
    {
        float m, v;
        localStats(src, x, y, m, v);
        int   Cp   = selectCp(v, V_max, S);
        float norm = static_cast<float>(Cp) + 12.f;

        float acc = static_cast<float>(Cp) * src.at(x, y);
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            acc += kW[dy+1][dx+1] * src.sample(x+dx, y+dy);
        }
        dst.at(x, y) = acc / norm;

        for (int ci = 0; ci < 6; ++ci)
            if (Cp == cpVals[ci]) { cpCount[ci]++; break; }
    }

    std::cout << "  [OLA diag] Cp distribution: ";
    for (int ci = 0; ci < 6; ++ci)
        std::cout << "Cp=" << cpVals[ci] << ":" << cpCount[ci] << " ";
    std::cout << std::endl;

    // Mean-preserving shift so H = G_LR - H_Ab is zero-mean
    float srcMean = src.mean();
    float dstMean = dst.mean();
    float shift   = srcMean - dstMean;
    std::cout << "  [OLA diag] LA mean correction: srcMean=" << srcMean
              << " blurMean=" << dstMean << " shift=" << shift << std::endl;
    for (float& v : dst.d)
        v = std::max(0.f, std::min(255.f, v + shift));

    return dst;
}

// =============================================================================
// SECTION 2 — HPF + USM  [Eq. 4, 1]
// =============================================================================

static FImg computeHPF(const FImg& G_LR, const FImg& H_Ab)
{
    FImg H(G_LR.w, G_LR.h);
    for (size_t i = 0; i < G_LR.d.size(); ++i)
        H.d[i] = G_LR.d[i] - H_Ab.d[i];
    return H;
}

static FImg applyUSM(const FImg& G_LR, const FImg& H, float k)
{
    FImg out(G_LR.w, G_LR.h);
    for (size_t i = 0; i < G_LR.d.size(); ++i)
        out.d[i] = std::max(0.f, std::min(255.f, G_LR.d[i] + k * H.d[i]));
    return out;
}

// =============================================================================
// SECTION 3 — CUCKOO SEARCH  [Algorithm 2]
//
// FINAL FITNESS DESIGN:
//
// The core problem: any sharpness-based fitness is nearly monotone for
// typical images because gradient energy increases with k until extreme
// clipping occurs. The clipping threshold depends heavily on image content.
//
// Solution: sample the fitness landscape at coarse intervals first to find
// the approximate peak k*, then run CS in the range [0, k*] to find the
// precise optimum. This guarantees CS finds an interior optimum.
//
// The fitness function itself uses a normalized sharpness measure:
//   fitness = (sharpness_gain / base_sharpness) / (1 + clip_penalty)
//
// Dividing by base_sharpness normalizes away image brightness dependency.
// clip_penalty = mean(excess^2) / (pixel_range^2) where pixel_range=255.
//
// This gives a dimensionless fitness that peaks at the optimal k.
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
    return 0.5*std::log(2.0*M_PI)+(x+0.5)*std::log(t)-t+std::log(a);
}
static double gammaFn(double x) { return std::exp(logGamma(x)); }

static double computeSigmaM(double beta)
{
    double num = std::sin(M_PI*beta/2.0)*gammaFn(1.0+beta);
    double den = gammaFn((1.0+beta)/2.0)*beta*std::pow(2.0,(beta-1.0)/2.0);
    return std::pow(num/den, 1.0/beta);
}

// Normalized fitness: sharpness gain ratio penalized by normalized clipping.
static double fitness(const FImg& G_LR, const FImg& H, float k, int step)
{
    double gradLRSum   = 0.0;  // base sharpness of G_LR
    double gradGain    = 0.0;  // sharpness gain from USM
    double clipExcess  = 0.0;  // clipping energy
    int    count       = 0;

    for (int y = 0; y < G_LR.h; y += step)
    for (int x = 0; x < G_LR.w; x += step)
    {
        // Base gradient
        float lrC  = G_LR.at(x,y);
        float lrR  = G_LR.sample(x+1,y);
        float lrD  = G_LR.sample(x,y+1);
        float gLR  = std::abs(lrR-lrC) + std::abs(lrD-lrC);
        gradLRSum += gLR;

        // USM raw values
        float raw  = lrC + k * H.at(x,y);
        float rawR = lrR + k * H.sample(x+1,y);
        float rawD = lrD + k * H.sample(x,y+1);

        // Clipping excess
        auto excess = [](float v) -> float {
            if (v > 255.f) return v - 255.f;
            if (v < 0.f)   return -v;
            return 0.f;
        };
        float ex = excess(raw) + excess(rawR) + excess(rawD);
        clipExcess += ex * ex;

        // Clamped gradient
        float val  = std::max(0.f, std::min(255.f, raw));
        float valR = std::max(0.f, std::min(255.f, rawR));
        float valD = std::max(0.f, std::min(255.f, rawD));
        float gSLR = std::abs(valR-val) + std::abs(valD-val);
        gradGain  += (gSLR - gLR);

        ++count;
    }

    if (count == 0 || gradLRSum < 1e-6) return 0.0;

    // Normalize sharpness gain by base sharpness (dimensionless)
    double normGain = gradGain / gradLRSum;

    // Normalize clipping by pixel range squared (dimensionless)
    double normClip = clipExcess / (count * 255.0 * 255.0);

    // Heavy clipping penalty
    return normGain / (1.0 + 50.0 * normClip);
}

// Scan k from 0 to kMax in nSteps, return k at peak fitness.
// Used to find the search range upper bound for CS.
static float findPeakK(const FImg& G_LR, const FImg& H,
                       float kMax, int nSteps, int step)
{
    float bestK   = 0.f;
    double bestF  = -1e30;
    std::cout << "  [OLA diag] Fitness scan: ";
    for (int i = 0; i <= nSteps; ++i)
    {
        float k = kMax * i / nSteps;
        double f = fitness(G_LR, H, k, step);
        std::cout << "k=" << std::fixed << std::setprecision(2) << k
                  << ":" << std::setprecision(2) << f << " ";
        if (f > bestF) { bestF = f; bestK = k; }
    }
    std::cout << std::endl;
    std::cout << "  [OLA diag] Peak at k=" << bestK
              << " fitness=" << bestF << std::endl;
    return bestK;
}

} // namespace cs

static float cuckoosearchOptimiseK(const FImg& G_LR, const FImg& H,
                                   const OLAESplineParams& p)
{
    const int    n    = p.csNests;
    const double beta = p.csBeta;
    const double sigM = cs::computeSigmaM(beta);

    // Step 1: Scan landscape at coarse resolution to find peak k
    // Search over full range [0, csKMax] in 20 steps
    float peakK = cs::findPeakK(G_LR, H, (float)p.csKMax, 20, p.csFitnessStep);

    // Step 2: CS searches in [0, peakK] — guaranteed interior optimum
    // Use peakK as upper bound; if peakK is at boundary, use 0.7*csKMax
    double csKMax = (peakK >= (float)p.csKMax * 0.95f)
                    ? p.csKMax * 0.7
                    : (double)peakK;
    double csKMin = 0.0;

    std::cout << "  [OLA diag] CS search range: [" << csKMin << ", "
              << csKMax << "]" << std::endl;
    std::cout << "  [OLA diag] CS: nests=" << n << " gen=" << p.csMaxGen
              << " beta=" << beta << " sigmaM=" << std::fixed
              << std::setprecision(4) << sigM << std::endl;

    std::mt19937                           rng(p.csSeed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double>       normM(0.0, sigM);
    std::normal_distribution<double>       normN(0.0, 1.0);

    std::vector<double> nests(n);
    for (auto& k : nests)
        k = csKMin + uni(rng) * (csKMax - csKMin);

    std::vector<double> fit(n);
    for (int i = 0; i < n; ++i)
        fit[i] = cs::fitness(G_LR, H, (float)nests[i], p.csFitnessStep);

    int bestIdx = (int)(std::max_element(fit.begin(), fit.end()) - fit.begin());

    std::cout << "  [OLA diag] CS init: best k=" << std::fixed
              << std::setprecision(4) << nests[bestIdx]
              << " fitness=" << fit[bestIdx] << std::endl;

    for (int t = 0; t < p.csMaxGen; ++t)
    {
        int i = std::min((int)(uni(rng) * n), n-1);

        double mi = normM(rng);
        double ni = normN(rng);
        if (std::abs(ni) < 1e-10) ni = 1e-10;

        double ratio = mi / ni;
        double s = 0.01 * std::pow(std::abs(ratio), 1.0/beta)
                        * (ratio < 0.0 ? -1.0 : 1.0);
        double levy = std::pow((double)(t+1), -beta);

        double newK = nests[i] + s * levy;
        newK = std::max(csKMin, std::min(csKMax, newK));

        double newFit = cs::fitness(G_LR, H, (float)newK, p.csFitnessStep);

        int j = std::min((int)(uni(rng) * n), n-1);
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
        for (int w2 = 0; w2 < abandon; ++w2)
        {
            int wi    = idx[w2];
            nests[wi] = csKMin + uni(rng) * (csKMax - csKMin);
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
// Replicate boundary throughout.
// =============================================================================

static float beta3(float t)
{
    float a = std::abs(t);
    if (a < 1.f) return 2.f/3.f - 0.5f*a*a*(2.f-a);
    if (a < 2.f) { float d=2.f-a; return d*d*d/6.f; }
    return 0.f;
}

static void bsplinePrefilter1D(std::vector<float>& c, int n)
{
    if (n < 2) return;
    const float z1     = std::sqrt(3.f) - 2.f;
    const float lambda = (1.f - z1) * (1.f - 1.f / z1);
    for (auto& v : c) v *= lambda;
    c[0]   = c[0] / (1.f - z1);
    for (int i = 1; i < n; ++i) c[i] += z1 * c[i-1];
    c[n-1] = c[n-1] * z1 / (z1 - 1.f);
    for (int i = n-2; i >= 0; --i) c[i] = z1 * (c[i+1] - c[i]);
}

static FImg computeBSplineCoeffs(const FImg& src)
{
    FImg C = src;
    std::vector<float> row(src.w);
    for (int y = 0; y < src.h; ++y)
    {
        for (int x = 0; x < src.w; ++x) row[x] = C.at(x,y);
        bsplinePrefilter1D(row, src.w);
        for (int x = 0; x < src.w; ++x) C.at(x,y) = row[x];
    }
    std::vector<float> col(src.h);
    for (int x = 0; x < src.w; ++x)
    {
        for (int y = 0; y < src.h; ++y) col[y] = C.at(x,y);
        bsplinePrefilter1D(col, src.h);
        for (int y = 0; y < src.h; ++y) C.at(x,y) = col[y];
    }
    return C;
}

static float evalBSpline(const FImg& C, float fx, float fy)
{
    int x0=(int)std::floor(fx), y0=(int)std::floor(fy);
    float acc=0.f;
    for (int j=y0-1; j<=y0+2; ++j)
    for (int i=x0-1; i<=x0+2; ++i)
        acc += C.sample(i,j)*beta3(fx-(float)i)*beta3(fy-(float)j);
    return acc;
}

static FImg bsplineUpscale(const FImg& G_SLR, int scale)
{
    FImg C = computeBSplineCoeffs(G_SLR);
    printImgStats("B-spline coefficients", C);

    int outW=G_SLR.w*scale, outH=G_SLR.h*scale;
    FImg G_HR(outW, outH);
    const float fs=(float)scale;
    for (int oy=0; oy<outH; ++oy)
    for (int ox=0; ox<outW;  ++ox)
    {
        float fx=(ox+0.5f)/fs-0.5f;
        float fy=(oy+0.5f)/fs-0.5f;
        G_HR.at(ox,oy)=std::max(0.f,std::min(255.f,evalBSpline(C,fx,fy)));
    }
    return G_HR;
}

// =============================================================================
// SECTION 5 — CANNY EDGE DETECTION
// =============================================================================

static FImg cannyEdgeDetect(const FImg& src, float lowT, float highT)
{
    int W=src.w, H=src.h;
    static const float gk[5][5]={
        {2,4,5,4,2},{4,9,12,9,4},{5,12,15,12,5},{4,9,12,9,4},{2,4,5,4,2}};
    const float gkSum=159.f;

    FImg sm(W,H);
    for(int y=0;y<H;++y) for(int x=0;x<W;++x){
        float acc=0.f;
        for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx)
            acc+=src.sample(x+dx,y+dy)*gk[dy+2][dx+2];
        sm.at(x,y)=acc/gkSum;
    }

    FImg mag(W,H),ang(W,H);
    for(int y=0;y<H;++y) for(int x=0;x<W;++x){
        float gx=-sm.sample(x-1,y-1)+sm.sample(x+1,y-1)
                 -2.f*sm.sample(x-1,y)+2.f*sm.sample(x+1,y)
                 -sm.sample(x-1,y+1)+sm.sample(x+1,y+1);
        float gy=-sm.sample(x-1,y-1)-2.f*sm.sample(x,y-1)
                 -sm.sample(x+1,y-1)+sm.sample(x-1,y+1)
                 +2.f*sm.sample(x,y+1)+sm.sample(x+1,y+1);
        mag.at(x,y)=std::sqrt(gx*gx+gy*gy);
        ang.at(x,y)=std::atan2(gy,gx);
    }

    FImg nms(W,H);
    for(int y=1;y<H-1;++y) for(int x=1;x<W-1;++x){
        float m=mag.at(x,y);
        float deg=ang.at(x,y)*180.f/(float)M_PI;
        if(deg<0.f) deg+=180.f;
        float m1,m2;
        if     (deg<22.5f||deg>=157.5f){m1=mag.at(x-1,y);  m2=mag.at(x+1,y);  }
        else if(deg<67.5f)             {m1=mag.at(x-1,y+1);m2=mag.at(x+1,y-1);}
        else if(deg<112.5f)            {m1=mag.at(x,y-1);  m2=mag.at(x,y+1);  }
        else                           {m1=mag.at(x-1,y-1);m2=mag.at(x+1,y+1);}
        nms.at(x,y)=(m>=m1&&m>=m2)?m:0.f;
    }

    FImg edges(W,H);
    for(int i=0;i<W*H;++i){
        float v=nms.d[i];
        edges.d[i]=(v>=highT)?255.f:(v>=lowT)?128.f:0.f;
    }
    for(int y=1;y<H-1;++y) for(int x=1;x<W-1;++x){
        if(edges.at(x,y)!=128.f) continue;
        bool conn=false;
        for(int dy=-1;dy<=1&&!conn;++dy) for(int dx=-1;dx<=1&&!conn;++dx)
            conn=(edges.at(x+dx,y+dy)==255.f);
        edges.at(x,y)=conn?255.f:0.f;
    }

    int ec=0; for(float v:edges.d) if(v>=255.f) ++ec;
    std::cout<<"  [OLA diag] Canny: "<<ec<<" edge pixels ("
             <<std::fixed<<std::setprecision(1)<<100.f*ec/(W*H)<<"%)"<<std::endl;
    return edges;
}

// =============================================================================
// SECTION 6 — e-SPLINE EDGE EXPANSION  [Eq. 9-15]
// =============================================================================

static FImg eSplineEdgeExpansion(const FImg& G_HR, const FImg& edgeMap)
{
    int W=G_HR.w, H=G_HR.h;
    FImg G_e(W,H);
    int mc=0;

    for(int y=2;y<H-2;++y) for(int x=2;x<W-2;++x){
        if(edgeMap.at(x,y)<128.f) continue;
        float SDh=
            0.25f*(G_HR.sample(x-1,y-1)-G_HR.sample(x-1,y+1))+
            0.50f*(G_HR.sample(x,  y-1)-G_HR.sample(x,  y+1))+
            0.25f*(G_HR.sample(x+1,y-1)-G_HR.sample(x+1,y+1));
        float SDv=
            0.25f*(G_HR.sample(x-1,y-1)-G_HR.sample(x+1,y-1))+
            0.50f*(G_HR.sample(x-1,y  )-G_HR.sample(x+1,y  ))+
            0.25f*(G_HR.sample(x-1,y+1)-G_HR.sample(x+1,y+1));

        if(std::abs(SDh)>=std::abs(SDv)){
            G_e.at(x,y-1)=0.5f*(G_HR.sample(x,y-1)+G_HR.sample(x,y-2))-G_HR.at(x,y-1);
            G_e.at(x,y+1)=0.5f*(G_HR.sample(x,y+1)+G_HR.sample(x,y+2))-G_HR.at(x,y+1);
        } else {
            G_e.at(x+1,y)=0.5f*(G_HR.sample(x+1,y)+G_HR.sample(x+2,y))-G_HR.at(x+1,y);
            G_e.at(x-1,y)=0.5f*(G_HR.sample(x-1,y)+G_HR.sample(x-2,y))-G_HR.at(x-1,y);
        }
        ++mc;
    }
    std::cout<<"  [OLA diag] Edge expansion: "<<mc<<" edge pixels modified"<<std::endl;

    FImg G_RHR(W,H);
    for(int i=0;i<W*H;++i)
        G_RHR.d[i]=std::max(0.f,std::min(255.f,G_HR.d[i]+G_e.d[i]));
    return G_RHR;
}

// =============================================================================
// SECTION 7 — SINGLE-CHANNEL PIPELINE
// =============================================================================

static FImg olaESplineSingleChannel(const FImg& G_LR, const OLAESplineParams& p)
{
    std::cout<<"  [OLA diag] --- Channel pipeline start ---"<<std::endl;
    printImgStats("G_LR input", G_LR);

    FImg H_Ab=laAdaptiveGaussianBlur(G_LR);
    printImgStats("H_Ab (mean-corrected blur)", H_Ab);

    FImg H=computeHPF(G_LR, H_Ab);
    printImgStats("H (HPF)", H);

    float k=cuckoosearchOptimiseK(G_LR, H, p);

    FImg G_SLR=applyUSM(G_LR, H, k);
    printImgStats("G_SLR (sharpened LR)", G_SLR);

    FImg G_HR=bsplineUpscale(G_SLR, p.scaleFactor);
    printImgStats("G_HR (B-spline upscaled)", G_HR);

    FImg edgeMap=cannyEdgeDetect(G_HR, p.cannyLow, p.cannyHigh);

    FImg G_RHR=eSplineEdgeExpansion(G_HR, edgeMap);
    printImgStats("G_RHR (final output)", G_RHR);

    std::cout<<"  [OLA diag] --- Channel pipeline end ---"<<std::endl;
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
    const int oW=inW*params.scaleFactor;
    const int oH=inH*params.scaleFactor;

    const char* chName[]={"R","G","B"};
    for(int ch=0;ch<3;++ch){
        std::cout<<"  [OLA] Processing channel "<<chName[ch]<<std::endl;
        FImg G_LR(inW,inH);
        for(int y=0;y<inH;++y) for(int x=0;x<inW;++x)
            G_LR.at(x,y)=(float)input[(y*inW+x)*4+ch];

        FImg G_RHR=olaESplineSingleChannel(G_LR,params);

        for(int y=0;y<oH;++y) for(int x=0;x<oW;++x){
            float v=G_RHR.at(x,y);
            output[(y*oW+x)*4+ch]=(unsigned char)(std::max(0.f,std::min(255.f,v))+0.5f);
        }
    }

    for(int oy=0;oy<oH;++oy) for(int ox=0;ox<oW;++ox){
        int sx=std::min(ox/params.scaleFactor,inW-1);
        int sy=std::min(oy/params.scaleFactor,inH-1);
        output[(oy*oW+ox)*4+3]=input[(sy*inW+sx)*4+3];
    }
}
