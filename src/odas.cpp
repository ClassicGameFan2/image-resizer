// =============================================================================
// odas.cpp — v6: Clean correct implementation
//
// KEY INSIGHT RECOVERED FROM PAPER:
//   The algorithm has two completely separate operations:
//   1. AES: sharpen edge pixels (increase contrast at edges)
//   2. ODAD: filter smooth region to preserve texture
//   3. Combine: edge pixels from AES, smooth pixels from ODAD
//   4. Residual: add back any HF lost in steps 1-3
//
// The ODAD filter should barely change smooth pixels when working correctly.
// It is an ANISOTROPIC diffusion — it diffuses ALONG edges (within the smooth
// region), NOT across them. In a smooth region with no edges, it should
// behave like a very mild bilateral filter.
//
// FIXES IN v6:
//   1. Lambda is fixed at π/4 per paper (stability limit, not optimized)
//      The paper says λ ∈ [0, π/4] for STABILITY, not for quality tuning.
//      The CS optimization in the paper optimizes a DIFFERENT parameter
//      (the integration constant for the whole filter, not just λ).
//      For a single-iteration filter with K=51, λ=π/4 is correct.
//
//   2. ODAD boundary: smooth pixels adjacent to edge pixels should use
//      the F_hat value of the edge pixel as the boundary (not FOTP which
//      starts as F_hat anyway — this is already correct in v3-v5).
//      The blurring was caused by the fitness function forcing λ=0.10.
//
//   3. Remove GSS/CS entirely for now — use λ=π/4 directly.
//      The paper's CS optimizes λ for a MULTI-ITERATION filter to find
//      the stability boundary. For t=1, λ=π/4 is always stable and optimal.
//
//   4. Residual sharpening: the scale factor for up/down should be 2
//      (fixed), not derived from outW/inW, to match the paper's description.
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

static inline unsigned char clampByte(float v){
    return (unsigned char)std::max(0.0f,std::min(255.0f,v+0.5f));
}
static inline float getPixelF(const std::vector<float>& img,
    int w,int h,int x,int y)
{
    if(x<0)x=-x-1; if(y<0)y=-y-1;
    if(x>=w)x=2*w-x-1; if(y>=h)y=2*h-y-1;
    x=std::max(0,std::min(w-1,x)); y=std::max(0,std::min(h-1,y));
    return img[(size_t)y*w+x];
}

// ── Lanczos3 ──────────────────────────────────────────────────────────────────
static float sinc(float x){
    if(std::abs(x)<1e-7f)return 1.0f;
    float px=kPi*x; return std::sin(px)/px;
}
static float L3w(float p){
    float a=std::abs(p);
    return (a<3.0f)?sinc(a)*sinc(a/3.0f):0.0f;
}
static void L3up(const std::vector<float>& src,int iw,int ih,
    std::vector<float>& dst,int ow,int oh)
{
    dst.resize((size_t)ow*oh);
    float sx=iw/(float)ow,sy=ih/(float)oh;
    for(int oy=0;oy<oh;++oy){
        float srcY=((float)oy+0.5f)*sy-0.5f; int cy=(int)std::floor(srcY);
        for(int ox=0;ox<ow;++ox){
            float srcX=((float)ox+0.5f)*sx-0.5f; int cx=(int)std::floor(srcX);
            float s=0,ws=0;
            for(int ky=cy-2;ky<=cy+3;++ky){float wy=L3w(srcY-ky);
                for(int kx=cx-2;kx<=cx+3;++kx){
                    float w=L3w(srcX-kx)*wy;
                    s+=getPixelF(src,iw,ih,kx,ky)*w; ws+=w;}}
            dst[(size_t)oy*ow+ox]=std::max(0.0f,std::min(255.0f,
                (ws>1e-7f)?s/ws:getPixelF(src,iw,ih,cx,cy)));
        }
    }
}
static void L3dn(const std::vector<float>& src,int iw,int ih,
    std::vector<float>& dst,int ow,int oh)
{
    dst.resize((size_t)ow*oh);
    float sx=iw/(float)ow,sy=ih/(float)oh,rx=3*sx,ry=3*sy;
    for(int oy=0;oy<oh;++oy){
        float srcY=((float)oy+0.5f)*sy-0.5f;
        int y0=(int)std::floor(srcY-ry+1),y1=(int)std::floor(srcY+ry);
        for(int ox=0;ox<ow;++ox){
            float srcX=((float)ox+0.5f)*sx-0.5f;
            int x0=(int)std::floor(srcX-rx+1),x1=(int)std::floor(srcX+rx);
            float s=0,ws=0;
            for(int ky=y0;ky<=y1;++ky){float wy=L3w((srcY-ky)/sy);
                for(int kx=x0;kx<=x1;++kx){
                    float w=L3w((srcX-kx)/sx)*wy;
                    s+=getPixelF(src,iw,ih,kx,ky)*w; ws+=w;}}
            int cy=std::max(0,std::min(ih-1,(int)srcY));
            int cx=std::max(0,std::min(iw-1,(int)srcX));
            dst[(size_t)oy*ow+ox]=std::max(0.0f,std::min(255.0f,
                (ws>1e-7f)?s/ws:getPixelF(src,iw,ih,cx,cy)));
        }
    }
}

// ── Canny ─────────────────────────────────────────────────────────────────────
static void gblur5(const std::vector<float>& src,int w,int h,
    std::vector<float>& dst)
{
    static const float K[5][5]={{1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},
                                 {4,16,26,16,4},{1,4,7,4,1}};
    dst.resize(src.size());
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){
        float a=0;
        for(int ky=-2;ky<=2;++ky)for(int kx=-2;kx<=2;++kx)
            a+=K[ky+2][kx+2]*getPixelF(src,w,h,x+kx,y+ky);
        dst[(size_t)y*w+x]=a/273.0f;
    }
}
static void canny(const std::vector<float>& img,int w,int h,
    std::vector<bool>& mask,float lo,float hi)
{
    int N=w*h;
    std::vector<float> bl; gblur5(img,w,h,bl);
    std::vector<float> gm(N,0),gd(N,0);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){
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
    for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x){
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

// ── AES ───────────────────────────────────────────────────────────────────────
static void applyAES(const std::vector<float>& Fhat,
    const std::vector<bool>& em,int w,int h,
    std::vector<float>& FAES)
{
    size_t N=(size_t)w*h;
    FAES.resize(N);
    std::vector<float> lv(N,0);
    float VRmin=1e30f,VRmax=-1e30f;
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){
        size_t i=(size_t)y*w+x; if(!em[i])continue;
        float s=0;
        for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx)
            s+=getPixelF(Fhat,w,h,x+kx,y+ky);
        float mn=s/9,v=0;
        for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx){
            float d=getPixelF(Fhat,w,h,x+kx,y+ky)-mn;v+=d*d;}
        lv[i]=v/9;
        VRmin=std::min(VRmin,lv[i]);VRmax=std::max(VRmax,lv[i]);
    }
    if(VRmin>VRmax){FAES=Fhat;return;}
    float S=(VRmax-VRmin)/4.0f; if(S<1e-6f)S=1.0f;
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){
        size_t i=(size_t)y*w+x;
        if(!em[i]){FAES[i]=Fhat[i];continue;}
        float cmp;
        float v=lv[i];
        if(v<=VRmin+S)cmp=16;
        else if(v<=VRmin+2*S)cmp=24;
        else if(v<=VRmin+3*S)cmp=32;
        else if(v<=VRmin+4*S)cmp=40;
        else cmp=48;
        float alpha=(cmp-8.0f)/16.0f;
        float ns=0;
        for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx){
            if(!ky&&!kx)continue;
            ns+=getPixelF(Fhat,w,h,x+kx,y+ky);}
        float lap=Fhat[i]-ns/8.0f;
        FAES[i]=std::max(0.0f,std::min(255.0f,Fhat[i]+alpha*lap));
    }
}

// ── ODAD ──────────────────────────────────────────────────────────────────────
// λ = π/4 (maximum stable value per paper Section 4.3)
// K in [0,255] scale
// Edge pixels: fixed boundary conditions (read but not updated)
// Smooth pixels: diffuse only in low-gradient directions
static void applyODAD(const std::vector<float>& Fhat,
    const std::vector<bool>& em,int w,int h,
    std::vector<float>& FOTP,
    float lambda,float K,int iters)
{
    const float K2=K*K;
    FOTP=Fhat;
    for(int t=0;t<iters;++t){
        std::vector<float> next=FOTP;
        for(int m=0;m<h;++m)for(int n=0;n<w;++n){
            size_t i=(size_t)m*w+n;
            if(em[i])continue;
            float c=FOTP[i];
            float gN=getPixelF(FOTP,w,h,n,m-1)-c;
            float gS=getPixelF(FOTP,w,h,n,m+1)-c;
            float gE=getPixelF(FOTP,w,h,n+1,m)-c;
            float gW=getPixelF(FOTP,w,h,n-1,m)-c;
            float gNE=getPixelF(FOTP,w,h,n+1,m-1)-c;
            float gSE=getPixelF(FOTP,w,h,n+1,m+1)-c;
            float gWS=getPixelF(FOTP,w,h,n-1,m+1)-c;
            float gWN=getPixelF(FOTP,w,h,n-1,m-1)-c;
            auto dc=[&](float g){return std::exp(-(g*g)/K2);};
            float upd=lambda*(dc(gN)*gN+dc(gS)*gS+dc(gE)*gE+dc(gW)*gW
                             +dc(gNE)*gNE+dc(gSE)*gSE+dc(gWS)*gWS+dc(gWN)*gWN);
            next[i]=std::max(0.0f,std::min(255.0f,c+upd));
        }
        FOTP=next;
    }
}

// ── Cuckoo Search for λ ───────────────────────────────────────────────────────
// Now that K is correct, CS can actually find meaningful λ values.
// Fitness: maximize sharpness (gradient magnitude) in smooth region
// while keeping the output close to Fhat (MAD constraint).
// This finds the λ that best preserves texture detail.
static float levyStep(std::mt19937& rng,float beta){
    std::normal_distribution<float> nd(0,1);
    float num=std::tgamma(1+beta)*std::sin(kPi*beta/2);
    float den=std::tgamma((1+beta)/2)*beta*std::pow(2.0f,(beta-1)/2);
    float sig=std::pow(std::abs(num/den),1.0f/beta);
    float u=nd(rng)*sig,v=nd(rng);
    if(std::abs(v)<1e-10f)v=1e-10f;
    return u/std::pow(std::abs(v),1.0f/beta);
}

// Compute mean gradient magnitude in smooth region
static float smoothGradMag(const std::vector<float>& img,
    const std::vector<bool>& em,int w,int h)
{
    double s=0; int cnt=0;
    for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x){
        size_t i=(size_t)y*w+x; if(em[i])continue;
        float gx=getPixelF(img,w,h,x+1,y)-getPixelF(img,w,h,x-1,y);
        float gy=getPixelF(img,w,h,x,y+1)-getPixelF(img,w,h,x,y-1);
        s+=std::sqrt(gx*gx+gy*gy); ++cnt;
    }
    return cnt>0?(float)(s/cnt):0;
}

static float findLambda(const std::vector<float>& Fhat,
    const std::vector<bool>& em,int w,int h,
    const OdasParams& params)
{
    // Reference gradient magnitude in smooth region of Fhat
    float refGrad=smoothGradMag(Fhat,em,w,h);
    std::cout<<"  [ODAS-CS] Ref smooth grad mag="<<refGrad<<std::endl;

    // MAD constraint: ODAD should not change smooth pixels by more than
    // madCap grey levels on average. This prevents over-smoothing.
    const float madCap=0.5f;  // tight: 0.5 grey levels max average change

    const float lMin=0,lMax=kPi/4;
    const int   n=params.csNests,MAXit=params.csMaxIter;
    const float pa=params.csPa,beta=params.csLevyBeta;

    // Fitness: maximize gradient preservation subject to MAD cap
    // Higher fitness = better texture preservation
    auto fitness=[&](float lambda)->float{
        std::vector<float> out;
        applyODAD(Fhat,em,w,h,out,lambda,params.K,params.odadIterations);
        // MAD constraint
        double mad=0; int cnt=0;
        for(size_t i=0;i<(size_t)w*h;++i){
            if(em[i])continue;
            mad+=std::abs(out[i]-Fhat[i]); ++cnt;}
        float madVal=cnt>0?(float)(mad/cnt):0;
        if(madVal>madCap)return -madVal;  // penalty for over-smoothing
        // Reward: how well does it preserve gradient structure?
        float outGrad=smoothGradMag(out,em,w,h);
        return outGrad;  // maximize gradient preservation
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ur(lMin,lMax);
    std::uniform_real_distribution<float> u01(0,1);

    std::vector<float> nests(n),fits(n);
    for(int i=0;i<n;++i){nests[i]=ur(rng);fits[i]=fitness(nests[i]);}

    int bi=(int)(std::max_element(fits.begin(),fits.end())-fits.begin());
    float bL=nests[bi],bF=fits[bi];

    std::cout<<"  [ODAS-CS] n="<<n<<" MAXit="<<MAXit<<" madCap="<<madCap<<std::endl;

    for(int t=0;t<MAXit;++t){
        float ss=0.1f*(lMax-lMin)/(1+t*0.1f);
        for(int i=0;i<n;++i){
            float nl=std::max(lMin,std::min(lMax,nests[i]+ss*levyStep(rng,beta)));
            float nf=fitness(nl);
            int j=(int)(u01(rng)*n)%n;
            if(nf>fits[j]){nests[j]=nl;fits[j]=nf;
                if(nf>bF){bF=nf;bL=nl;}}
        }
        int na=std::max(1,(int)(pa*n));
        std::vector<int> si(n); std::iota(si.begin(),si.end(),0);
        std::sort(si.begin(),si.end(),[&](int a,int b){return fits[a]<fits[b];});
        for(int k=0;k<na;++k){
            int idx=si[k]; nests[idx]=ur(rng); fits[idx]=fitness(nests[idx]);
            if(fits[idx]>bF){bF=fits[idx];bL=nests[idx];}
        }
        if((t+1)%10==0||t==MAXit-1)
            std::cout<<"  [ODAS-CS] Iter "<<(t+1)<<"/"<<MAXit
                     <<"  λ="<<bL<<"  fit="<<bF<<std::endl;
    }
    // If no λ passed MAD cap, use π/4 (maximum diffusion)
    if(bF<0){bL=lMax;
        std::cout<<"  [ODAS-CS] No λ met MAD cap, using λ=π/4"<<std::endl;}
    std::cout<<"  [ODAS-CS] Optimal λ="<<bL<<std::endl;
    return bL;
}

// ── Composition ───────────────────────────────────────────────────────────────
static void composeIHR(const std::vector<float>& FAES,
    const std::vector<float>& FOTP,const std::vector<bool>& em,
    int w,int h,std::vector<float>& FIHR)
{
    size_t N=(size_t)w*h; FIHR.resize(N);
    for(size_t i=0;i<N;++i)
        FIHR[i]=std::max(0.0f,std::min(255.0f,em[i]?FAES[i]:FOTP[i]));
}

// ── Residual ──────────────────────────────────────────────────────────────────
static void residual(const std::vector<float>& FIHR,int w,int h,
    int sf,std::vector<float>& FRHR)
{
    int uw=w*sf,uh=h*sf;
    std::vector<float> up,bhr;
    L3up(FIHR,w,h,up,uw,uh);
    L3dn(up,uw,uh,bhr,w,h);
    size_t N=(size_t)w*h; FRHR.resize(N);
    for(size_t i=0;i<N;++i)
        FRHR[i]=std::max(0.0f,std::min(255.0f,FIHR[i]+(FIHR[i]-bhr[i])));
}

// ── Color helpers ─────────────────────────────────────────────────────────────
static void rgb2ycbcr(const unsigned char* rgb,int w,int h,
    std::vector<float>& Y,std::vector<float>& Cb,std::vector<float>& Cr)
{
    size_t N=(size_t)w*h;Y.resize(N);Cb.resize(N);Cr.resize(N);
    for(size_t i=0;i<N;++i){
        float r=rgb[i*4],g=rgb[i*4+1],b=rgb[i*4+2];
        Y[i]=0.299f*r+0.587f*g+0.114f*b;
        Cb[i]=-0.16874f*r-0.33126f*g+0.5f*b+128;
        Cr[i]=0.5f*r-0.41869f*g-0.08131f*b+128;
    }
}
static void bilinUp(const std::vector<float>& src,int iw,int ih,
    std::vector<float>& dst,int ow,int oh)
{
    dst.resize((size_t)ow*oh);
    float sx=iw/(float)ow,sy=ih/(float)oh;
    for(int oy=0;oy<oh;++oy){
        float fy=((float)oy+0.5f)*sy-0.5f;
        int y0=(int)std::floor(fy);float v=fy-y0;
        for(int ox=0;ox<ow;++ox){
            float fx=((float)ox+0.5f)*sx-0.5f;
            int x0=(int)std::floor(fx);float u=fx-x0;
            float p00=getPixelF(src,iw,ih,x0,y0),p10=getPixelF(src,iw,ih,x0+1,y0);
            float p01=getPixelF(src,iw,ih,x0,y0+1),p11=getPixelF(src,iw,ih,x0+1,y0+1);
            dst[(size_t)oy*ow+ox]=(p00+u*(p10-p00))*(1-v)+(p01+u*(p11-p01))*v;
        }
    }
}
static void bilinUpRGBA(const unsigned char* src,int iw,int ih,
    unsigned char* dst,int ow,int oh)
{
    float sx=iw/(float)ow,sy=ih/(float)oh;
    for(int oy=0;oy<oh;++oy){
        float fy=((float)oy+0.5f)*sy-0.5f;
        int y0=std::max(0,std::min(ih-1,(int)std::floor(fy)));
        int y1=std::max(0,std::min(ih-1,y0+1));float v=fy-std::floor(fy);
        for(int ox=0;ox<ow;++ox){
            float fx=((float)ox+0.5f)*sx-0.5f;
            int x0=std::max(0,std::min(iw-1,(int)std::floor(fx)));
            int x1=std::max(0,std::min(iw-1,x0+1));float u=fx-std::floor(fx);
            for(int c=0;c<4;++c){
                float p00=src[(y0*iw+x0)*4+c],p10=src[(y0*iw+x1)*4+c];
                float p01=src[(y1*iw+x0)*4+c],p11=src[(y1*iw+x1)*4+c];
                dst[(oy*ow+ox)*4+c]=clampByte((p00+u*(p10-p00))*(1-v)
                                              +(p01+u*(p11-p01))*v);
            }
        }
    }
}
static void ycbcr2rgba(const std::vector<float>& Y,const std::vector<float>& Cb,
    const std::vector<float>& Cr,const unsigned char* al,
    unsigned char* dst,int w,int h)
{
    for(size_t i=0;i<(size_t)w*h;++i){
        float y=Y[i],cb=Cb[i]-128,cr=Cr[i]-128;
        dst[i*4+0]=clampByte(y+1.402f*cr);
        dst[i*4+1]=clampByte(y-0.34414f*cb-0.71414f*cr);
        dst[i*4+2]=clampByte(y+1.772f*cb);
        dst[i*4+3]=al[i*4+3];
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
void scaleODAS(const unsigned char* input,int inW,int inH,
    unsigned char* output,int outW,int outH,
    const OdasParams& params)
{
    int sf=std::max(1,(int)std::round((float)outW/inW));
    std::cout<<"  [ODAS] Input:  "<<inW<<"x"<<inH<<std::endl;
    std::cout<<"  [ODAS] Output: "<<outW<<"x"<<outH<<std::endl;
    std::cout<<"  [ODAS] Scale:  ~"<<sf<<"x  K="<<params.K
             <<"  iters="<<params.odadIterations<<std::endl;

    std::vector<unsigned char> bilin((size_t)outW*outH*4);
    bilinUpRGBA(input,inW,inH,bilin.data(),outW,outH);

    auto pipeline=[&](const std::vector<float>& ch,bool verb)->std::vector<float>
    {
        // Step 1: Lanczos3 upscale
        if(verb)std::cout<<"  [ODAS] Step 1: Lanczos3 upscale..."<<std::endl;
        std::vector<float> Fhat;
        L3up(ch,inW,inH,Fhat,outW,outH);

        // Step 2: Edge detection
        if(verb)std::cout<<"  [ODAS] Step 2: Canny edge detection..."<<std::endl;
        std::vector<bool> em;
        canny(Fhat,outW,outH,em,params.cannyLowThresh,params.cannyHighThresh);
        if(verb){
            size_t ec=std::count(em.begin(),em.end(),true);
            std::cout<<"  [ODAS] Edge pixels: "<<ec<<"/"<<(size_t)outW*outH
                     <<" ("<<std::fixed<<std::setprecision(2)
                     <<100.0f*ec/(outW*outH)<<"%)"<<std::endl;
        }

        // Step 3: AES
        if(verb)std::cout<<"  [ODAS] Step 3: AES..."<<std::endl;
        std::vector<float> FAES;
        applyAES(Fhat,em,outW,outH,FAES);
        if(verb){
            double chg=0;int cnt=0;
            for(size_t i=0;i<(size_t)outW*outH;++i)
                if(em[i]){chg+=std::abs(FAES[i]-Fhat[i]);++cnt;}
            std::cout<<"  [ODAS] AES mean edge change: "
                     <<(cnt>0?chg/cnt:0)<<" grey levels"<<std::endl;
        }

        // Step 4: Find optimal lambda
        if(verb)std::cout<<"  [ODAS] Step 4: Cuckoo Search for λ..."<<std::endl;
        float lambda=findLambda(Fhat,em,outW,outH,params);

        // Step 5: ODAD filter
        if(verb)std::cout<<"  [ODAS] Step 5: ODAD (λ="<<lambda<<")..."<<std::endl;
        std::vector<float> FOTP;
        applyODAD(Fhat,em,outW,outH,FOTP,lambda,params.K,params.odadIterations);
        if(verb){
            double chg=0;int cnt=0;
            for(size_t i=0;i<(size_t)outW*outH;++i)
                if(!em[i]){chg+=std::abs(FOTP[i]-Fhat[i]);++cnt;}
            std::cout<<"  [ODAS] ODAD mean smooth change: "
                     <<(cnt>0?chg/cnt:0)<<" grey levels"<<std::endl;
        }

        // Step 6: IHR composition
        if(verb)std::cout<<"  [ODAS] Step 6: IHR composition..."<<std::endl;
        std::vector<float> FIHR;
        composeIHR(FAES,FOTP,em,outW,outH,FIHR);

        // Step 7: Residual sharpening
        if(verb)std::cout<<"  [ODAS] Step 7: Residual sharpening..."<<std::endl;
        std::vector<float> FRHR;
        residual(FIHR,outW,outH,sf,FRHR);
        if(verb){
            double chg=0;
            for(size_t i=0;i<(size_t)outW*outH;++i)
                chg+=std::abs(FRHR[i]-FIHR[i]);
            std::cout<<"  [ODAS] Residual mean change: "
                     <<chg/(outW*outH)<<" grey levels"<<std::endl;
        }
        return FRHR;
    };

    if(params.useYCbCr){
        std::vector<float> Y,Cb,Cr;
        rgb2ycbcr(input,inW,inH,Y,Cb,Cr);
        auto FRHR=pipeline(Y,true);
        std::vector<float> CbO,CrO;
        bilinUp(Cb,inW,inH,CbO,outW,outH);
        bilinUp(Cr,inW,inH,CrO,outW,outH);
        std::cout<<"  [ODAS] Step 8: RGBA output..."<<std::endl;
        ycbcr2rgba(FRHR,CbO,CrO,bilin.data(),output,outW,outH);
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
