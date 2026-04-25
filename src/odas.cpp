// =============================================================================
// odas.cpp  — v4: Fixed AES (unsharp masking), fixed CS fitness
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
// SECTION 1: Lanczos3 (unchanged)
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
            float sum = 0, wSum = 0;
            for (int ky = cy-2; ky <= cy+3; ++ky) {
                float wy = lanczos3Weight(srcY - (float)ky);
                for (int kx = cx-2; kx <= cx+3; ++kx) {
                    float w = lanczos3Weight(srcX - (float)kx) * wy;
                    sum  += getPixelF(src, inW, inH, kx, ky) * w;
                    wSum += w;
                }
            }
            float val = (wSum > 1e-7f) ? sum/wSum : getPixelF(src,inW,inH,cx,cy);
            dst[(size_t)oy * outW + ox] = std::max(0.0f, std::min(255.0f, val));
        }
    }
}

static void lanczos3Downscale(
    const std::vector<float>& src, int inW, int inH,
    std::vector<float>&       dst, int outW, int outH)
{
    dst.resize((size_t)outW * outH);
    float scaleX = (float)inW/(float)outW;
    float scaleY = (float)inH/(float)outH;
    float radX = 3.0f*scaleX, radY = 3.0f*scaleY;
    for (int oy = 0; oy < outH; ++oy) {
        float srcY = ((float)oy+0.5f)*scaleY-0.5f;
        int yMin=(int)std::floor(srcY-radY+1), yMax=(int)std::floor(srcY+radY);
        for (int ox = 0; ox < outW; ++ox) {
            float srcX = ((float)ox+0.5f)*scaleX-0.5f;
            int xMin=(int)std::floor(srcX-radX+1), xMax=(int)std::floor(srcX+radX);
            float sum=0, wSum=0;
            for (int ky=yMin;ky<=yMax;++ky) {
                float wy=lanczos3Weight((srcY-(float)ky)/scaleY);
                for (int kx=xMin;kx<=xMax;++kx) {
                    float w=lanczos3Weight((srcX-(float)kx)/scaleX)*wy;
                    sum+=getPixelF(src,inW,inH,kx,ky)*w; wSum+=w;
                }
            }
            int cy=std::max(0,std::min(inH-1,(int)srcY));
            int cx=std::max(0,std::min(inW-1,(int)srcX));
            float val=(wSum>1e-7f)?sum/wSum:getPixelF(src,inW,inH,cx,cy);
            dst[(size_t)oy*outW+ox]=std::max(0.0f,std::min(255.0f,val));
        }
    }
}

// =============================================================================
// SECTION 2: Canny Edge Detection (unchanged)
// =============================================================================

static void gaussianBlur5(const std::vector<float>& src, int w, int h,
                           std::vector<float>& dst)
{
    static const float K[5][5]={
        {1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1}};
    dst.resize(src.size());
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        float acc=0;
        for (int ky=-2;ky<=2;++ky) for (int kx=-2;kx<=2;++kx)
            acc+=K[ky+2][kx+2]*getPixelF(src,w,h,x+kx,y+ky);
        dst[(size_t)y*w+x]=acc/273.0f;
    }
}

static void cannyEdgeDetect(
    const std::vector<float>& img, int w, int h,
    std::vector<bool>& edgeMask, float lowT, float highT)
{
    int N=w*h;
    std::vector<float> blurred;
    gaussianBlur5(img,w,h,blurred);

    std::vector<float> gMag(N,0), gDir(N,0);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        float gx=
            -getPixelF(blurred,w,h,x-1,y-1)+getPixelF(blurred,w,h,x+1,y-1)+
            -2*getPixelF(blurred,w,h,x-1,y)+2*getPixelF(blurred,w,h,x+1,y)+
            -getPixelF(blurred,w,h,x-1,y+1)+getPixelF(blurred,w,h,x+1,y+1);
        float gy=
            -getPixelF(blurred,w,h,x-1,y-1)-2*getPixelF(blurred,w,h,x,y-1)+
            -getPixelF(blurred,w,h,x+1,y-1)+getPixelF(blurred,w,h,x-1,y+1)+
            2*getPixelF(blurred,w,h,x,y+1)+getPixelF(blurred,w,h,x+1,y+1);
        size_t i=(size_t)y*w+x;
        gMag[i]=std::sqrt(gx*gx+gy*gy);
        gDir[i]=std::atan2(gy,gx);
    }

    std::vector<float> sup(N,0);
    for (int y=1;y<h-1;++y) for (int x=1;x<w-1;++x) {
        size_t i=(size_t)y*w+x;
        float mag=gMag[i], ang=gDir[i]*180.0f/kPi;
        if (ang<0) ang+=180;
        float m1,m2;
        if      (ang<22.5f||ang>=157.5f){m1=gMag[i+1];m2=gMag[i-1];}
        else if (ang<67.5f)             {m1=gMag[i-(size_t)w+1];m2=gMag[i+(size_t)w-1];}
        else if (ang<112.5f)            {m1=gMag[i-(size_t)w];m2=gMag[i+(size_t)w];}
        else                            {m1=gMag[i-(size_t)w-1];m2=gMag[i+(size_t)w+1];}
        sup[i]=(mag>=m1&&mag>=m2)?mag:0;
    }

    std::vector<int> st(N,0);
    for (int i=0;i<N;++i){
        if(sup[i]>=highT)st[i]=2;
        else if(sup[i]>=lowT)st[i]=1;
    }
    std::vector<int> stk; stk.reserve(N/8);
    for (int i=0;i<N;++i) if(st[i]==2) stk.push_back(i);
    while (!stk.empty()) {
        int idx=stk.back(); stk.pop_back();
        int px=idx%w, py=idx/w;
        for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
            if(!dx&&!dy) continue;
            int nx=px+dx, ny=py+dy;
            if(nx<0||nx>=w||ny<0||ny>=h) continue;
            int ni=ny*w+nx;
            if(st[ni]==1){st[ni]=2;stk.push_back(ni);}
        }
    }
    edgeMask.assign(N,false);
    for (int i=0;i<N;++i) edgeMask[i]=(st[i]==2);
}

// =============================================================================
// SECTION 3: AES — v4 FIX: Unsharp Masking (not raw Laplacian output)
//
// THE BUG IN v1/v2/v3:
//   result = cmp*center + sum(neighbours*(-cmp/8))
//          = cmp*(center - mean_neighbours)
//   This is the RAW LAPLACIAN — an edge detector output.
//   For cmp=48: values range roughly [-48*255, +48*255] = [-12240, +12240]
//   After clamping to [0,255]: dark side of edge → 0 (BLACK), bright → 255 (WHITE)
//
// THE FIX: Unsharp Masking = original + scaled_laplacian
//   laplacian  = center - mean_neighbours           [local detail signal]
//   sharpened  = center + alpha * laplacian         [add detail back]
//
//   where alpha is derived from cmp so that:
//     cmp=16 → alpha=0.5   (mild sharpening)
//     cmp=24 → alpha=1.0
//     cmp=32 → alpha=1.5
//     cmp=40 → alpha=2.0
//     cmp=48 → alpha=2.5   (strong sharpening)
//
//   These alpha values keep the output well within [0,255] for typical
//   image content (laplacian magnitude is usually small).
//   Final clamp to [0,255] handles extreme cases safely.
//
// WHY THIS IS CORRECT:
//   The paper says the kernel H performs "sharpening" — unsharp masking
//   is THE standard Laplacian sharpening technique. The kernel H with
//   sum=0 IS a Laplacian kernel. Sharpening = image + c*Laplacian(image).
//   The cmp value controls the sharpening strength c.
// =============================================================================

static void applyAES(
    const std::vector<float>& Fhat,
    const std::vector<bool>&  edgeMask,
    int w, int h,
    std::vector<float>&       FAES)
{
    const size_t N = (size_t)w * h;
    FAES.resize(N);

    // Compute local variance at edge pixels (from Fhat neighbors)
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

    float VRmin=1e30f, VRmax=-1e30f;
    for (size_t i=0;i<N;++i) {
        if (!edgeMask[i]) continue;
        VRmin=std::min(VRmin,localVar[i]);
        VRmax=std::max(VRmax,localVar[i]);
    }
    if (VRmin > VRmax) { FAES = Fhat; return; }

    float S = (VRmax - VRmin) / 4.0f;
    if (S < 1e-6f) S = 1.0f;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t idx = (size_t)y*w+x;

            if (!edgeMask[idx]) {
                FAES[idx] = Fhat[idx];  // smooth pixels pass through unchanged
                continue;
            }

            float lvr = localVar[idx];

            // cmp selects sharpening strength (from paper's Algorithm 1)
            float cmp;
            if      (lvr <= VRmin +       S) cmp = 16.0f;
            else if (lvr <= VRmin + 2.0f*S)  cmp = 24.0f;
            else if (lvr <= VRmin + 3.0f*S)  cmp = 32.0f;
            else if (lvr <= VRmin + 4.0f*S)  cmp = 40.0f;
            else                              cmp = 48.0f;

            // Convert cmp to unsharp masking alpha
            // cmp=16→α=0.5, cmp=24→α=1.0, cmp=32→α=1.5, cmp=40→α=2.0, cmp=48→α=2.5
            float alpha = (cmp - 8.0f) / 16.0f;

            // Compute 8-neighbour mean (mean of neighbours, excluding center)
            float neighbourSum = 0.0f;
            for (int ky=-1;ky<=1;++ky) for (int kx=-1;kx<=1;++kx) {
                if (!ky && !kx) continue;
                neighbourSum += getPixelF(Fhat,w,h,x+kx,y+ky);
            }
            float neighbourMean = neighbourSum / 8.0f;

            // Laplacian = center - neighbour_mean  (local detail signal)
            float center    = getPixelF(Fhat, w, h, x, y);
            float laplacian = center - neighbourMean;

            // Unsharp masking: sharpened = center + alpha * laplacian
            float sharpened = center + alpha * laplacian;

            FAES[idx] = std::max(0.0f, std::min(255.0f, sharpened));
        }
    }
}

// =============================================================================
// SECTION 4: ODAD Filter (v3 version — boundary conditions correct)
// =============================================================================

static void applyODAD(
    const std::vector<float>& Fhat,
    const std::vector<bool>&  edgeMask,
    int w, int h,
    std::vector<float>&       FOTP,
    float lambda, float K, int iterations)
{
    const size_t N = (size_t)w * h;
    const float  K2 = K * K;
    FOTP = Fhat;  // edge pixels fixed, smooth pixels diffuse

    for (int t = 0; t < iterations; ++t) {
        std::vector<float> next = FOTP;
        for (int m = 0; m < h; ++m) {
            for (int n = 0; n < w; ++n) {
                size_t idx = (size_t)m*w+n;
                if (edgeMask[idx]) continue;  // fixed boundary

                float c = FOTP[idx];
                float gN =getPixelF(FOTP,w,h,n,  m-1)-c;
                float gS =getPixelF(FOTP,w,h,n,  m+1)-c;
                float gE =getPixelF(FOTP,w,h,n+1,m  )-c;
                float gW =getPixelF(FOTP,w,h,n-1,m  )-c;
                float gNE=getPixelF(FOTP,w,h,n+1,m-1)-c;
                float gSE=getPixelF(FOTP,w,h,n+1,m+1)-c;
                float gWS=getPixelF(FOTP,w,h,n-1,m+1)-c;
                float gWN=getPixelF(FOTP,w,h,n-1,m-1)-c;

                auto dc=[&](float g){return std::exp(-(g*g)/K2);};

                float upd=lambda*(
                    dc(gN)*gN+dc(gS)*gS+dc(gE)*gE+dc(gW)*gW+
                    dc(gNE)*gNE+dc(gSE)*gSE+dc(gWS)*gWS+dc(gWN)*gWN);

                next[idx]=std::max(0.0f,std::min(255.0f,c+upd));
            }
        }
        FOTP=next;
    }
}

// =============================================================================
// SECTION 5: Cuckoo Search — v4 FIX: Correct fitness function
//
// PROBLEM WITH v3 "maximize Laplacian energy":
//   Laplacian energy always increases with λ because larger λ means
//   MORE diffusion update which perturbs pixels MORE → more local
//   contrast variation → higher Laplacian → CS always picks λ=π/4.
//
// THE CORRECT FITNESS:
//   We want ODAD to PRESERVE smooth region texture while NOT bleeding
//   across edges. The ideal λ minimizes the difference between
//   ODAD(Fhat) and Fhat at smooth pixel locations.
//
//   fitness(λ) = -MSE(ODAD(Fhat, λ), Fhat)  [at smooth pixels only]
//              = -(1/N) * Σ_{smooth i} (ODAD_i - Fhat_i)²
//
//   Maximize fitness = minimize distortion from Fhat.
//   The optimal λ is the LARGEST value that keeps distortion below
//   a threshold — meaning: as much diffusion as possible without
//   visibly altering the smooth region texture.
//
//   In practice, since t=1, the distortion grows smoothly with λ.
//   The CS will find the λ that sits at the "knee" of the curve.
//
// IMPLEMENTATION NOTE:
//   We negate MSE so that maximizing fitness = minimizing distortion.
//   Starting fitness at λ=0 is 0 (no change = perfect preservation).
//   We want the λ that maximizes texture-directed diffusion while
//   keeping MSE below a perceptual threshold (~0.5 in [0,255] space).
//
//   Better formulation: maximize diffusion benefit while constraining
//   distortion. We implement this as a constrained fitness:
//
//   fitness(λ) = λ  if  MSE(ODAD(λ), Fhat) < threshold
//              = -∞  otherwise
//
//   threshold = 1.0  (1 grey level² average distortion, imperceptible)
//
//   This finds the LARGEST λ that doesn't visibly distort smooth pixels.
// =============================================================================

static float smoothMSE(
    const std::vector<float>& filtered,
    const std::vector<float>& reference,
    const std::vector<bool>&  edgeMask,
    int w, int h)
{
    double sum = 0.0;
    int    cnt = 0;
    for (size_t i = 0; i < (size_t)w*h; ++i) {
        if (edgeMask[i]) continue;
        double d = filtered[i] - reference[i];
        sum += d * d;
        ++cnt;
    }
    return (cnt > 0) ? (float)(sum / cnt) : 0.0f;
}

static float levyStep(std::mt19937& rng, float beta) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float num  = std::tgamma(1.0f+beta)*std::sin(kPi*beta/2.0f);
    float den  = std::tgamma((1.0f+beta)/2.0f)*beta*
                 std::pow(2.0f,(beta-1.0f)/2.0f);
    float sig  = std::pow(std::abs(num/den), 1.0f/beta);
    float u    = nd(rng)*sig;
    float v    = nd(rng);
    if (std::abs(v)<1e-10f) v=1e-10f;
    return u/std::pow(std::abs(v),1.0f/beta);
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

    // Distortion threshold: 1.0 grey-level² MSE is imperceptible
    // (equivalent to ~0.4% of full range, far below visible threshold)
    const float mseThreshold = 1.0f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniRange(lambdaMin, lambdaMax);
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);

    // Fitness: largest λ whose ODAD distortion stays below threshold.
    // Returns λ if acceptable, -1 if distortion too high.
    // We maximize this, so CS will find the largest acceptable λ.
    auto fitness = [&](float lambda) -> float {
        std::vector<float> filtered;
        applyODAD(Fhat, edgeMask, w, h, filtered,
                  lambda, params.K, params.odadIterations);
        float mse = smoothMSE(filtered, Fhat, edgeMask, w, h);
        // Return λ itself if within threshold (maximize λ subject to constraint)
        // Return negative penalty if over threshold
        return (mse <= mseThreshold) ? lambda : (-mse);
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
              << "  Pa=" << pa << "  MSE_threshold=" << mseThreshold
              << std::endl;

    for (int t = 0; t < MAXit; ++t) {
        float stepScale = 0.01f*(lambdaMax-lambdaMin)/(1.0f+(float)t*0.05f);

        for (int i = 0; i < n; ++i) {
            float newL = std::max(lambdaMin,
                         std::min(lambdaMax,
                                  nests[i]+stepScale*levyStep(rng,beta)));
            float newF = fitness(newL);
            int j = (int)(uni01(rng)*(float)n)%n;
            if (newF > fitnesses[j]) {
                nests[j]=newL; fitnesses[j]=newF;
                if (newF > bestFit) { bestFit=newF; bestLambda=newL; }
            }
        }

        // Abandon worst nests
        int numAbandon = std::max(1,(int)(pa*(float)n));
        std::vector<int> si(n);
        std::iota(si.begin(),si.end(),0);
        std::sort(si.begin(),si.end(),
                  [&](int a,int b){return fitnesses[a]<fitnesses[b];});
        for (int k=0;k<numAbandon;++k) {
            int idx=si[k];
            nests[idx]=uniRange(rng);
            fitnesses[idx]=fitness(nests[idx]);
            if (fitnesses[idx]>bestFit){bestFit=fitnesses[idx];bestLambda=nests[idx];}
        }

        if ((t+1)%20==0 || t==MAXit-1) {
            // Compute actual MSE for reporting
            std::vector<float> tmp;
            applyODAD(Fhat,edgeMask,w,h,tmp,bestLambda,params.K,params.odadIterations);
            float mse=smoothMSE(tmp,Fhat,edgeMask,w,h);
            std::cout << "  [ODAS-CS] Iter " << (t+1) << "/" << MAXit
                      << "  Best λ=" << bestLambda
                      << "  MSE=" << mse << std::endl;
        }
    }

    // Safety: if no λ passed the threshold, use a small safe default
    if (bestFit < 0.0f) {
        bestLambda = 0.05f;
        std::cout << "  [ODAS-CS] Warning: no λ met threshold, using default 0.05"
                  << std::endl;
    }

    std::cout << "  [ODAS-CS] Optimal λ = " << bestLambda << std::endl;
    return bestLambda;
}

// =============================================================================
// SECTION 6: IHR Composition (unchanged from v3)
// =============================================================================

static void composeIHR(
    const std::vector<float>& FAES,
    const std::vector<float>& FOTP,
    const std::vector<bool>&  edgeMask,
    int w, int h,
    std::vector<float>&       FIHR)
{
    const size_t N = (size_t)w*h;
    FIHR.resize(N);
    for (size_t i=0;i<N;++i)
        FIHR[i]=std::max(0.0f,std::min(255.0f,
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
    int upW=w*scaleFactor, upH=h*scaleFactor;
    std::vector<float> upscaled, FBHR;
    lanczos3Upscale(FIHR,w,h,upscaled,upW,upH);
    lanczos3Downscale(upscaled,upW,upH,FBHR,w,h);
    const size_t N=(size_t)w*h;
    FRHR.resize(N);
    for (size_t i=0;i<N;++i)
        FRHR[i]=std::max(0.0f,std::min(255.0f,FIHR[i]+(FIHR[i]-FBHR[i])));
}

// =============================================================================
// SECTION 8: Color space helpers (unchanged)
// =============================================================================

static void rgbToYCbCr(const unsigned char* rgb, int w, int h,
    std::vector<float>& Y, std::vector<float>& Cb, std::vector<float>& Cr)
{
    const size_t N=(size_t)w*h;
    Y.resize(N); Cb.resize(N); Cr.resize(N);
    for (size_t i=0;i<N;++i) {
        float r=rgb[i*4],g=rgb[i*4+1],b=rgb[i*4+2];
        Y[i] = 0.299f*r+0.587f*g+0.114f*b;
        Cb[i]=-0.16874f*r-0.33126f*g+0.5f*b+128.0f;
        Cr[i]= 0.5f*r-0.41869f*g-0.08131f*b+128.0f;
    }
}

static void bilinearUpscale(const std::vector<float>& src, int inW, int inH,
    std::vector<float>& dst, int outW, int outH)
{
    dst.resize((size_t)outW*outH);
    float sx=(float)inW/(float)outW, sy=(float)inH/(float)outH;
    for (int oy=0;oy<outH;++oy) {
        float fy=((float)oy+0.5f)*sy-0.5f;
        int y0=(int)std::floor(fy); float v=fy-(float)y0;
        for (int ox=0;ox<outW;++ox) {
            float fx=((float)ox+0.5f)*sx-0.5f;
            int x0=(int)std::floor(fx); float u=fx-(float)x0;
            dst[(size_t)oy*outW+ox]=
                (getPixelF(src,inW,inH,x0,y0)+u*(getPixelF(src,inW,inH,x0+1,y0)
                -getPixelF(src,inW,inH,x0,y0)))*(1-v)+
                (getPixelF(src,inW,inH,x0,y0+1)+u*(getPixelF(src,inW,inH,x0+1,y0+1)
                -getPixelF(src,inW,inH,x0,y0+1)))*v;
        }
    }
}

static void bilinearUpscaleRGBA(const unsigned char* src, int inW, int inH,
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
                float p00=src[(y0*inW+x0)*4+c],p10=src[(y0*inW+x1)*4+c];
                float p01=src[(y1*inW+x0)*4+c],p11=src[(y1*inW+x1)*4+c];
                dst[(oy*outW+ox)*4+c]=clampByte(
                    (p00+u*(p10-p00))*(1-v)+(p01+u*(p11-p01))*v);
            }
        }
    }
}

static void yCbCrToRgba(const std::vector<float>& Y,
    const std::vector<float>& Cb, const std::vector<float>& Cr,
    const unsigned char* alpha, unsigned char* dst, int w, int h)
{
    for (size_t i=0;i<(size_t)w*h;++i) {
        float y=Y[i],cb=Cb[i]-128,cr=Cr[i]-128;
        dst[i*4+0]=clampByte(y+1.40200f*cr);
        dst[i*4+1]=clampByte(y-0.34414f*cb-0.71414f*cr);
        dst[i*4+2]=clampByte(y+1.77200f*cb);
        dst[i*4+3]=alpha[i*4+3];
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
    int scaleFactor=std::max(1,(int)std::round((float)outW/(float)inW));

    std::cout<<"  [ODAS] Input:  "<<inW<<"x"<<inH<<std::endl;
    std::cout<<"  [ODAS] Output: "<<outW<<"x"<<outH<<std::endl;
    std::cout<<"  [ODAS] Scale factor: ~"<<scaleFactor<<"x"<<std::endl;
    std::cout<<"  [ODAS] K="<<params.K
             <<"  ODAD iterations="<<params.odadIterations<<std::endl;

    std::vector<unsigned char> bilinearOut((size_t)outW*outH*4);
    bilinearUpscaleRGBA(input,inW,inH,bilinearOut.data(),outW,outH);

    auto runPipeline = [&](const std::vector<float>& chanIn,
                           int chIdx) -> std::vector<float>
    {
        // Step 1: Lanczos3 upscale
        if (chIdx == 0)
            std::cout<<"  [ODAS] Step 1: Lanczos3 upscale..."<<std::endl;
        std::vector<float> Fhat;
        lanczos3Upscale(chanIn, inW, inH, Fhat, outW, outH);

        // Step 2: Canny edge detection
        if (chIdx == 0)
            std::cout<<"  [ODAS] Step 2: Canny edge detection..."<<std::endl;
        std::vector<bool> edgeMask;
        cannyEdgeDetect(Fhat, outW, outH, edgeMask,
                        params.cannyLowThresh, params.cannyHighThresh);

        if (chIdx == 0) {
            size_t ec=std::count(edgeMask.begin(),edgeMask.end(),true);
            std::cout<<"  [ODAS] Edge pixels: "<<ec
                     <<" / "<<((size_t)outW*outH)
                     <<" ("<<(100.0f*ec/(outW*outH))<<"%)"<<std::endl;
        }

        // Step 3: AES (unsharp masking at edge pixels)
        if (chIdx == 0)
            std::cout<<"  [ODAS] Step 3: Adaptive Edge Sharpening..."<<std::endl;
        std::vector<float> FAES;
        applyAES(Fhat, edgeMask, outW, outH, FAES);

        // Step 4: CS optimization for λ
        if (chIdx == 0)
            std::cout<<"  [ODAS] Step 4: Cuckoo Search for λ..."<<std::endl;
        float lambda = cuckooSearchLambda(Fhat, edgeMask, outW, outH, params);

        // Step 5: ODAD filter (edge pixels fixed as boundary conditions)
        if (chIdx == 0)
            std::cout<<"  [ODAS] Step 5: ODAD filter (λ="<<lambda<<")..."<<std::endl;
        std::vector<float> FOTP;
        applyODAD(Fhat, edgeMask, outW, outH, FOTP,
                  lambda, params.K, params.odadIterations);

        // Step 6: IHR masked merge
        if (chIdx == 0)
            std::cout<<"  [ODAS] Step 6: IHR composition..."<<std::endl;
        std::vector<float> FIHR;
        composeIHR(FAES, FOTP, edgeMask, outW, outH, FIHR);

        // Step 7: Residual sharpening
        if (chIdx == 0)
            std::cout<<"  [ODAS] Step 7: Residual sharpening..."<<std::endl;
        std::vector<float> FRHR;
        residualSharpening(FIHR, outW, outH, scaleFactor, FRHR);

        return FRHR;
    };

    if (params.useYCbCr) {
        std::vector<float> Yin, Cbin, Crin;
        rgbToYCbCr(input, inW, inH, Yin, Cbin, Crin);

        std::vector<float> FRHR = runPipeline(Yin, 0);

        std::vector<float> CbOut, CrOut;
        bilinearUpscale(Cbin, inW, inH, CbOut, outW, outH);
        bilinearUpscale(Crin, inW, inH, CrOut, outW, outH);

        std::cout<<"  [ODAS] Step 8: Compositing RGBA output..."<<std::endl;
        yCbCrToRgba(FRHR, CbOut, CrOut, bilinearOut.data(), output, outW, outH);

    } else {
        std::cout<<"  [ODAS] RGB mode: 3 channels"<<std::endl;
        std::vector<std::vector<float>> ch(3);
        for (int c=0;c<3;++c) {
            std::vector<float> chanIn((size_t)inW*inH);
            for (int i=0;i<inW*inH;++i) chanIn[i]=(float)input[i*4+c];
            ch[c] = runPipeline(chanIn, c);
        }
        for (int i=0;i<outW*outH;++i) {
            output[i*4+0]=clampByte(ch[0][i]);
            output[i*4+1]=clampByte(ch[1][i]);
            output[i*4+2]=clampByte(ch[2][i]);
            output[i*4+3]=bilinearOut[i*4+3];
        }
    }
    std::cout<<"  [ODAS] Done."<<std::endl;
}
