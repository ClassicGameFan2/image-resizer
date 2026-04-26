// =============================================================================
// odas.cpp  — v5: Fixed K scaling, simplified CS, verified pipeline
// 
// KEY FIXES:
//   1. K is now in [0,255] scale (paper uses [0,1], we use [0,255])
//      Default K=0.2 in paper → K=51.0 in our implementation
//      This is set in odas.h default: K=51.0f
//
//   2. CS replaced with a simple golden-section line search on λ.
//      The CS was never converging because ODAD barely changes anything
//      when K is wrong. A line search is more robust and faster.
//
//   3. Added diagnostic output to verify each pipeline stage is doing
//      something measurable.
//
//   4. AES alpha capped to avoid over-sharpening.
// =============================================================================
#include "odas.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>

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

// Compute mean absolute difference between two images at smooth pixels
static float smoothMAD(const std::vector<float>& a,
                       const std::vector<float>& b,
                       const std::vector<bool>&  edgeMask,
                       int w, int h)
{
    double sum = 0.0;
    int    cnt = 0;
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        if (edgeMask[i]) continue;
        sum += std::abs(a[i] - b[i]);
        ++cnt;
    }
    return (cnt > 0) ? (float)(sum / cnt) : 0.0f;
}

// =============================================================================
// SECTION 1: Lanczos3
// =============================================================================

static float sinc(float x) {
    if (std::abs(x) < 1e-7f) return 1.0f;
    float px = kPi * x;
    return std::sin(px) / px;
}
static float lanczos3W(float p) {
    float ap = std::abs(p);
    return (ap < 3.0f) ? sinc(ap) * sinc(ap / 3.0f) : 0.0f;
}

static void lanczos3Upscale(
    const std::vector<float>& src, int inW, int inH,
    std::vector<float>&       dst, int outW, int outH)
{
    dst.resize((size_t)outW * outH);
    float sx = (float)inW / outW, sy = (float)inH / outH;
    for (int oy = 0; oy < outH; ++oy) {
        float srcY = ((float)oy + 0.5f) * sy - 0.5f;
        int   cy   = (int)std::floor(srcY);
        for (int ox = 0; ox < outW; ++ox) {
            float srcX = ((float)ox + 0.5f) * sx - 0.5f;
            int   cx   = (int)std::floor(srcX);
            float sum = 0, ws = 0;
            for (int ky = cy-2; ky <= cy+3; ++ky) {
                float wy = lanczos3W(srcY - ky);
                for (int kx = cx-2; kx <= cx+3; ++kx) {
                    float w = lanczos3W(srcX - kx) * wy;
                    sum += getPixelF(src,inW,inH,kx,ky) * w;
                    ws  += w;
                }
            }
            dst[(size_t)oy*outW+ox] = std::max(0.0f, std::min(255.0f,
                (ws > 1e-7f) ? sum/ws : getPixelF(src,inW,inH,cx,cy)));
        }
    }
}

static void lanczos3Downscale(
    const std::vector<float>& src, int inW, int inH,
    std::vector<float>&       dst, int outW, int outH)
{
    dst.resize((size_t)outW * outH);
    float sx=inW/(float)outW, sy=inH/(float)outH;
    float rx=3.0f*sx, ry=3.0f*sy;
    for (int oy=0;oy<outH;++oy) {
        float srcY=((float)oy+0.5f)*sy-0.5f;
        int y0=(int)std::floor(srcY-ry+1), y1=(int)std::floor(srcY+ry);
        for (int ox=0;ox<outW;++ox) {
            float srcX=((float)ox+0.5f)*sx-0.5f;
            int x0=(int)std::floor(srcX-rx+1), x1=(int)std::floor(srcX+rx);
            float sum=0,ws=0;
            for (int ky=y0;ky<=y1;++ky) {
                float wy=lanczos3W((srcY-ky)/sy);
                for (int kx=x0;kx<=x1;++kx) {
                    float w=lanczos3W((srcX-kx)/sx)*wy;
                    sum+=getPixelF(src,inW,inH,kx,ky)*w; ws+=w;
                }
            }
            int cy=std::max(0,std::min(inH-1,(int)srcY));
            int cx=std::max(0,std::min(inW-1,(int)srcX));
            dst[(size_t)oy*outW+ox]=std::max(0.0f,std::min(255.0f,
                (ws>1e-7f)?sum/ws:getPixelF(src,inW,inH,cx,cy)));
        }
    }
}

// =============================================================================
// SECTION 2: Canny Edge Detection
// =============================================================================

static void gaussianBlur5(const std::vector<float>& src, int w, int h,
                           std::vector<float>& dst)
{
    static const float K[5][5]={
        {1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1}};
    dst.resize(src.size());
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        float a=0;
        for (int ky=-2;ky<=2;++ky) for (int kx=-2;kx<=2;++kx)
            a+=K[ky+2][kx+2]*getPixelF(src,w,h,x+kx,y+ky);
        dst[(size_t)y*w+x]=a/273.0f;
    }
}

static void cannyEdgeDetect(
    const std::vector<float>& img, int w, int h,
    std::vector<bool>& mask, float lo, float hi)
{
    int N=w*h;
    std::vector<float> bl; gaussianBlur5(img,w,h,bl);
    std::vector<float> gm(N,0),gd(N,0);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        float gx=-getPixelF(bl,w,h,x-1,y-1)+getPixelF(bl,w,h,x+1,y-1)
                 -2*getPixelF(bl,w,h,x-1,y)+2*getPixelF(bl,w,h,x+1,y)
                 -getPixelF(bl,w,h,x-1,y+1)+getPixelF(bl,w,h,x+1,y+1);
        float gy=-getPixelF(bl,w,h,x-1,y-1)-2*getPixelF(bl,w,h,x,y-1)
                 -getPixelF(bl,w,h,x+1,y-1)+getPixelF(bl,w,h,x-1,y+1)
                 +2*getPixelF(bl,w,h,x,y+1)+getPixelF(bl,w,h,x+1,y+1);
        size_t i=(size_t)y*w+x;
        gm[i]=std::sqrt(gx*gx+gy*gy); gd[i]=std::atan2(gy,gx);
    }
    std::vector<float> sup(N,0);
    for (int y=1;y<h-1;++y) for (int x=1;x<w-1;++x) {
        size_t i=(size_t)y*w+x;
        float m=gm[i],a=gd[i]*180/kPi; if(a<0)a+=180;
        float m1,m2;
        if(a<22.5f||a>=157.5f){m1=gm[i+1];m2=gm[i-1];}
        else if(a<67.5f){m1=gm[i-(size_t)w+1];m2=gm[i+(size_t)w-1];}
        else if(a<112.5f){m1=gm[i-(size_t)w];m2=gm[i+(size_t)w];}
        else{m1=gm[i-(size_t)w-1];m2=gm[i+(size_t)w+1];}
        sup[i]=(m>=m1&&m>=m2)?m:0;
    }
    std::vector<int> st(N,0);
    for(int i=0;i<N;++i){
        if(sup[i]>=hi)st[i]=2; else if(sup[i]>=lo)st[i]=1;}
    std::vector<int> stk; stk.reserve(N/8);
    for(int i=0;i<N;++i)if(st[i]==2)stk.push_back(i);
    while(!stk.empty()){
        int idx=stk.back();stk.pop_back();
        int px=idx%w,py=idx/w;
        for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
            if(!dx&&!dy)continue;
            int nx=px+dx,ny=py+dy;
            if(nx<0||nx>=w||ny<0||ny>=h)continue;
            int ni=ny*w+nx;
            if(st[ni]==1){st[ni]=2;stk.push_back(ni);}
        }
    }
    mask.assign(N,false);
    for(int i=0;i<N;++i)mask[i]=(st[i]==2);
}

// =============================================================================
// SECTION 3: AES — Unsharp Masking at edge pixels
// =============================================================================

static void applyAES(
    const std::vector<float>& Fhat,
    const std::vector<bool>&  edgeMask,
    int w, int h,
    std::vector<float>&       FAES)
{
    const size_t N = (size_t)w*h;
    FAES.resize(N);

    // Local variance at edge pixels
    std::vector<float> lv(N,0);
    float VRmin=1e30f, VRmax=-1e30f;
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        size_t i=(size_t)y*w+x;
        if (!edgeMask[i]) continue;
        float s=0;
        for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx)
            s+=getPixelF(Fhat,w,h,x+kx,y+ky);
        float mn=s/9,v=0;
        for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx){
            float d=getPixelF(Fhat,w,h,x+kx,y+ky)-mn; v+=d*d;}
        lv[i]=v/9;
        VRmin=std::min(VRmin,lv[i]); VRmax=std::max(VRmax,lv[i]);
    }
    if (VRmin>VRmax){FAES=Fhat;return;}
    float S=(VRmax-VRmin)/4.0f; if(S<1e-6f)S=1.0f;

    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        size_t i=(size_t)y*w+x;
        if (!edgeMask[i]){FAES[i]=Fhat[i];continue;}

        // Select sharpening strength from variance quartile
        float cmp;
        float v=lv[i];
        if     (v<=VRmin+  S)cmp=16;
        else if(v<=VRmin+2*S)cmp=24;
        else if(v<=VRmin+3*S)cmp=32;
        else if(v<=VRmin+4*S)cmp=40;
        else                  cmp=48;

        // Unsharp masking: output = center + alpha*(center - neighbour_mean)
        // alpha: cmp=16→0.5, 24→1.0, 32→1.5, 40→2.0, 48→2.5
        float alpha=(cmp-8.0f)/16.0f;

        float ns=0;
        for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx){
            if(!ky&&!kx)continue;
            ns+=getPixelF(Fhat,w,h,x+kx,y+ky);
        }
        float nmean=ns/8.0f;
        float center=Fhat[i];
        float lap=center-nmean;  // positive = brighter than surroundings
        float sharpened=center+alpha*lap;

        FAES[i]=std::max(0.0f,std::min(255.0f,sharpened));
    }
}

// =============================================================================
// SECTION 4: ODAD Filter
//
// CRITICAL FIX v5: K must be in [0,255] scale.
// The paper normalizes images to [0,1]. We work in [0,255].
// K=0.2 in paper = K=0.2*255=51.0 in our space.
// With K=0.2 on [0,255] data: exp(-(g/0.2)²) ≈ 0 for ANY g>0.5
// → effectively zero diffusion everywhere → no visible effect.
//
// The OdasParams default in odas.h MUST be changed to K=51.0f.
// =============================================================================

static void applyODAD(
    const std::vector<float>& Fhat,
    const std::vector<bool>&  edgeMask,
    int w, int h,
    std::vector<float>&       FOTP,
    float lambda, float K, int iterations)
{
    const float K2 = K * K;
    FOTP = Fhat;

    for (int t=0;t<iterations;++t) {
        std::vector<float> next=FOTP;
        for (int m=0;m<h;++m) for (int n=0;n<w;++n) {
            size_t i=(size_t)m*w+n;
            if (edgeMask[i]) continue;  // fixed boundary
            float c=FOTP[i];
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
            next[i]=std::max(0.0f,std::min(255.0f,c+upd));
        }
        FOTP=next;
    }
}

// =============================================================================
// SECTION 5: Lambda optimization — Golden Section Search
//
// Replace Cuckoo Search with Golden Section Search (GSS).
//
// WHY GSS INSTEAD OF CS:
//   The fitness landscape for λ is unimodal (single peak) for this problem:
//   - Too small λ: weak diffusion, output ≈ Fhat, low benefit
//   - Optimal λ:   diffuses noise in smooth regions, preserves edges
//   - Too large λ: over-smooths, diverges toward edge values
//
//   GSS finds the optimum of a unimodal function in O(log(1/ε)) evaluations.
//   CS is designed for multimodal landscapes and wastes evaluations here.
//
// FITNESS FUNCTION:
//   We want λ that MAXIMIZES diffusion benefit in smooth regions while
//   STAYING BELOW a distortion cap.
//
//   Benefit = how much the filter smooths uniform-gradient regions
//           = reduction in local variance in flat smooth areas
//   Cap     = max allowed MAD from Fhat (preserve overall structure)
//
//   fitness(λ) = variance_reduction(λ)   if MAD(λ) ≤ cap
//              = -∞                       otherwise
//
//   variance_reduction = var(Fhat_smooth) - var(ODAD_smooth)
//   Positive = filter reduced variance = smoothed noise = good
//
// PARAMETERS:
//   csMaxIter is reused as the number of GSS iterations (20-50 is sufficient)
//   MAD cap = 2.0 grey levels (imperceptible, keeps structure intact)
// =============================================================================

static float smoothVariance(const std::vector<float>& img,
                             const std::vector<bool>&  edgeMask,
                             int w, int h)
{
    // Compute mean of smooth pixels
    double sum=0; int cnt=0;
    for (size_t i=0;i<(size_t)w*h;++i){
        if(edgeMask[i])continue;
        sum+=img[i]; ++cnt;
    }
    if(cnt==0)return 0;
    float mn=(float)(sum/cnt);
    // Variance
    double v=0;
    for (size_t i=0;i<(size_t)w*h;++i){
        if(edgeMask[i])continue;
        float d=img[i]-mn; v+=d*d;
    }
    return (float)(v/cnt);
}

static float findOptimalLambda(
    const std::vector<float>& Fhat,
    const std::vector<bool>&  edgeMask,
    int w, int h,
    const OdasParams& params)
{
    const float lMin=0.0f, lMax=kPi/4.0f;
    const float madCap=2.0f;   // max 2 grey levels average distortion
    const int   iters=params.csMaxIter;  // reuse as GSS iterations

    // Pre-compute reference variance of smooth region in Fhat
    float varRef = smoothVariance(Fhat, edgeMask, w, h);

    std::cout<<"  [ODAS-CS] Ref smooth variance="<<varRef<<std::endl;

    // Fitness: variance reduction subject to MAD constraint
    auto fitness=[&](float lambda)->float{
        std::vector<float> filtered;
        applyODAD(Fhat,edgeMask,w,h,filtered,lambda,params.K,params.odadIterations);
        float mad=smoothMAD(filtered,Fhat,edgeMask,w,h);
        if(mad>madCap) return -1e9f;
        float varOut=smoothVariance(filtered,edgeMask,w,h);
        return varRef-varOut;  // positive = noise reduced
    };

    // Golden Section Search on [lMin, lMax]
    const float phi=(1.0f+std::sqrt(5.0f))/2.0f;
    const float resphi=2.0f-phi;

    float a=lMin, b=lMax;
    float x1=a+resphi*(b-a);
    float x2=b-resphi*(b-a);
    float f1=fitness(x1), f2=fitness(x2);

    float bestLambda=x1, bestFit=f1;

    std::cout<<"  [ODAS-CS] Golden Section Search, "<<iters<<" iters"<<std::endl;

    for(int i=0;i<iters;++i){
        if(f1<f2){
            a=x1; x1=x2; f1=f2;
            x2=b-resphi*(b-a); f2=fitness(x2);
            if(f2>bestFit){bestFit=f2;bestLambda=x2;}
        } else {
            b=x2; x2=x1; f2=f1;
            x1=a+resphi*(b-a); f1=fitness(x1);
            if(f1>bestFit){bestFit=f1;bestLambda=x1;}
        }
        if((i+1)%10==0||i==iters-1){
            std::cout<<"  [ODAS-CS] Iter "<<(i+1)<<"/"<<iters
                     <<"  λ="<<bestLambda
                     <<"  VarReduction="<<bestFit<<std::endl;
        }
    }

    // Safety: if no valid λ found (all over MAD cap), use conservative default
    if(bestFit<0){
        bestLambda=0.1f;
        std::cout<<"  [ODAS-CS] MAD cap too tight, using λ=0.1"<<std::endl;
    }

    std::cout<<"  [ODAS-CS] Optimal λ="<<bestLambda
             <<"  VarReduction="<<bestFit<<std::endl;
    return bestLambda;
}

// =============================================================================
// SECTION 6: IHR Composition
// =============================================================================

static void composeIHR(
    const std::vector<float>& FAES,
    const std::vector<float>& FOTP,
    const std::vector<bool>&  edgeMask,
    int w, int h,
    std::vector<float>&       FIHR)
{
    const size_t N=(size_t)w*h;
    FIHR.resize(N);
    for(size_t i=0;i<N;++i)
        FIHR[i]=std::max(0.0f,std::min(255.0f,
            edgeMask[i]?FAES[i]:FOTP[i]));
}

// =============================================================================
// SECTION 7: Residual Sharpening
// =============================================================================

static void residualSharpening(
    const std::vector<float>& FIHR, int w, int h,
    int scaleFactor, std::vector<float>& FRHR)
{
    int uw=w*scaleFactor, uh=h*scaleFactor;
    std::vector<float> up,bhr;
    lanczos3Upscale(FIHR,w,h,up,uw,uh);
    lanczos3Downscale(up,uw,uh,bhr,w,h);
    const size_t N=(size_t)w*h;
    FRHR.resize(N);
    for(size_t i=0;i<N;++i)
        FRHR[i]=std::max(0.0f,std::min(255.0f,FIHR[i]+(FIHR[i]-bhr[i])));
}

// =============================================================================
// SECTION 8: Color space helpers
// =============================================================================

static void rgbToYCbCr(const unsigned char* rgb, int w, int h,
    std::vector<float>& Y,std::vector<float>& Cb,std::vector<float>& Cr)
{
    size_t N=(size_t)w*h; Y.resize(N);Cb.resize(N);Cr.resize(N);
    for(size_t i=0;i<N;++i){
        float r=rgb[i*4],g=rgb[i*4+1],b=rgb[i*4+2];
        Y[i]=0.299f*r+0.587f*g+0.114f*b;
        Cb[i]=-0.16874f*r-0.33126f*g+0.5f*b+128;
        Cr[i]=0.5f*r-0.41869f*g-0.08131f*b+128;
    }
}
static void bilinearUpscale(const std::vector<float>& src,int iw,int ih,
    std::vector<float>& dst,int ow,int oh)
{
    dst.resize((size_t)ow*oh);
    float sx=iw/(float)ow,sy=ih/(float)oh;
    for(int oy=0;oy<oh;++oy){
        float fy=((float)oy+0.5f)*sy-0.5f;
        int y0=(int)std::floor(fy); float v=fy-y0;
        for(int ox=0;ox<ow;++ox){
            float fx=((float)ox+0.5f)*sx-0.5f;
            int x0=(int)std::floor(fx); float u=fx-x0;
            float p00=getPixelF(src,iw,ih,x0,y0),p10=getPixelF(src,iw,ih,x0+1,y0);
            float p01=getPixelF(src,iw,ih,x0,y0+1),p11=getPixelF(src,iw,ih,x0+1,y0+1);
            dst[(size_t)oy*ow+ox]=(p00+u*(p10-p00))*(1-v)+(p01+u*(p11-p01))*v;
        }
    }
}
static void bilinearUpscaleRGBA(const unsigned char* src,int iw,int ih,
    unsigned char* dst,int ow,int oh)
{
    float sx=iw/(float)ow,sy=ih/(float)oh;
    for(int oy=0;oy<oh;++oy){
        float fy=((float)oy+0.5f)*sy-0.5f;
        int y0=std::max(0,std::min(ih-1,(int)std::floor(fy)));
        int y1=std::max(0,std::min(ih-1,y0+1)); float v=fy-std::floor(fy);
        for(int ox=0;ox<ow;++ox){
            float fx=((float)ox+0.5f)*sx-0.5f;
            int x0=std::max(0,std::min(iw-1,(int)std::floor(fx)));
            int x1=std::max(0,std::min(iw-1,x0+1)); float u=fx-std::floor(fx);
            for(int c=0;c<4;++c){
                float p00=src[(y0*iw+x0)*4+c],p10=src[(y0*iw+x1)*4+c];
                float p01=src[(y1*iw+x0)*4+c],p11=src[(y1*iw+x1)*4+c];
                dst[(oy*ow+ox)*4+c]=clampByte((p00+u*(p10-p00))*(1-v)+(p01+u*(p11-p01))*v);
            }
        }
    }
}
static void yCbCrToRgba(const std::vector<float>& Y,const std::vector<float>& Cb,
    const std::vector<float>& Cr,const unsigned char* alpha,unsigned char* dst,int w,int h)
{
    for(size_t i=0;i<(size_t)w*h;++i){
        float y=Y[i],cb=Cb[i]-128,cr=Cr[i]-128;
        dst[i*4+0]=clampByte(y+1.402f*cr);
        dst[i*4+1]=clampByte(y-0.34414f*cb-0.71414f*cr);
        dst[i*4+2]=clampByte(y+1.772f*cb);
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
    int sf=std::max(1,(int)std::round((float)outW/inW));
    std::cout<<"  [ODAS] Input:  "<<inW<<"x"<<inH<<std::endl;
    std::cout<<"  [ODAS] Output: "<<outW<<"x"<<outH<<std::endl;
    std::cout<<"  [ODAS] Scale:  ~"<<sf<<"x"<<std::endl;
    std::cout<<"  [ODAS] K="<<params.K
             <<"  iterations="<<params.odadIterations<<std::endl;

    std::vector<unsigned char> bilin((size_t)outW*outH*4);
    bilinearUpscaleRGBA(input,inW,inH,bilin.data(),outW,outH);

    auto pipeline=[&](const std::vector<float>& ch, bool verbose)
        ->std::vector<float>
    {
        if(verbose)std::cout<<"  [ODAS] Step 1: Lanczos3 upscale..."<<std::endl;
        std::vector<float> Fhat;
        lanczos3Upscale(ch,inW,inH,Fhat,outW,outH);

        if(verbose)std::cout<<"  [ODAS] Step 2: Canny edge detection..."<<std::endl;
        std::vector<bool> em;
        cannyEdgeDetect(Fhat,outW,outH,em,params.cannyLowThresh,params.cannyHighThresh);
        if(verbose){
            size_t ec=std::count(em.begin(),em.end(),true);
            std::cout<<"  [ODAS] Edge pixels: "<<ec<<"/"<<(outW*outH)
                     <<" ("<<std::fixed<<std::setprecision(2)
                     <<100.0f*ec/(outW*outH)<<"%)"<<std::endl;
        }

        if(verbose)std::cout<<"  [ODAS] Step 3: AES..."<<std::endl;
        std::vector<float> FAES;
        applyAES(Fhat,em,outW,outH,FAES);

        // Diagnostic: how much did AES change edge pixels?
        if(verbose){
            double aesChange=0; int cnt=0;
            for(size_t i=0;i<(size_t)outW*outH;++i)
                if(em[i]){aesChange+=std::abs(FAES[i]-Fhat[i]);++cnt;}
            std::cout<<"  [ODAS] AES mean edge change: "
                     <<(cnt>0?aesChange/cnt:0)<<" grey levels"<<std::endl;
        }

        if(verbose)std::cout<<"  [ODAS] Step 4: Lambda optimization..."<<std::endl;
        float lambda=findOptimalLambda(Fhat,em,outW,outH,params);

        if(verbose)std::cout<<"  [ODAS] Step 5: ODAD (λ="<<lambda<<")..."<<std::endl;
        std::vector<float> FOTP;
        applyODAD(Fhat,em,outW,outH,FOTP,lambda,params.K,params.odadIterations);

        // Diagnostic: how much did ODAD change smooth pixels?
        if(verbose){
            double odadChange=0; int cnt=0;
            for(size_t i=0;i<(size_t)outW*outH;++i)
                if(!em[i]){odadChange+=std::abs(FOTP[i]-Fhat[i]);++cnt;}
            std::cout<<"  [ODAS] ODAD mean smooth change: "
                     <<(cnt>0?odadChange/cnt:0)<<" grey levels"<<std::endl;
        }

        if(verbose)std::cout<<"  [ODAS] Step 6: IHR composition..."<<std::endl;
        std::vector<float> FIHR;
        composeIHR(FAES,FOTP,em,outW,outH,FIHR);

        if(verbose)std::cout<<"  [ODAS] Step 7: Residual sharpening..."<<std::endl;
        std::vector<float> FRHR;
        residualSharpening(FIHR,outW,outH,sf,FRHR);

        // Diagnostic: how much did residual step change things?
        if(verbose){
            double resChange=0;
            for(size_t i=0;i<(size_t)outW*outH;++i)
                resChange+=std::abs(FRHR[i]-FIHR[i]);
            std::cout<<"  [ODAS] Residual mean change: "
                     <<resChange/(outW*outH)<<" grey levels"<<std::endl;
        }

        return FRHR;
    };

    if(params.useYCbCr){
        std::vector<float> Y,Cb,Cr;
        rgbToYCbCr(input,inW,inH,Y,Cb,Cr);
        auto FRHR=pipeline(Y,true);
        std::vector<float> CbO,CrO;
        bilinearUpscale(Cb,inW,inH,CbO,outW,outH);
        bilinearUpscale(Cr,inW,inH,CrO,outW,outH);
        std::cout<<"  [ODAS] Step 8: RGBA output..."<<std::endl;
        yCbCrToRgba(FRHR,CbO,CrO,bilin.data(),output,outW,outH);
    } else {
        std::cout<<"  [ODAS] RGB mode"<<std::endl;
        std::vector<std::vector<float>> ch(3);
        for(int c=0;c<3;++c){
            std::vector<float> ci((size_t)inW*inH);
            for(int i=0;i<inW*inH;++i)ci[i]=input[i*4+c];
            ch[c]=pipeline(ci,c==0);
        }
        for(int i=0;i<outW*outH;++i){
            output[i*4+0]=clampByte(ch[0][i]);
            output[i*4+1]=clampByte(ch[1][i]);
            output[i*4+2]=clampByte(ch[2][i]);
            output[i*4+3]=bilin[i*4+3];
        }
    }
    std::cout<<"  [ODAS] Done."<<std::endl;
}
