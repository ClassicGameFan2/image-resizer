// ICBI - Iterative Curvature Based Interpolation
// C++ port of icbi.m by Andrea Giachetti and Nicola Asuni
// Original MATLAB: GNU GPL v2
// Port faithfully follows the MATLAB algorithm structure.

#include "icbi.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>

// -------------------------------------------------------------------------
// Single-channel ICBI doubling pass.
// IMG is a flat row-major double array of size [m rows x n cols].
// Returns the expanded image of size [(2m-1) rows x (2n-1) cols].
// Directly mirrors the ZF loop body in icbi.m.
// -------------------------------------------------------------------------
static std::vector<double> icbiDoubleChannel(
    const std::vector<double>& IMG,
    int m, int n,          // m = rows, n = cols
    int pf, bool vr,
    int st, double tm, double tc,
    int sc, double ts,
    double al, double bt, double gm,
    bool fcbiOnly)
{
    int mm = 2 * m - 1;   // expanded rows
    int nn = 2 * n - 1;   // expanded cols

    // 0-based flat indexers
    auto IDX  = [&](int i, int j) -> int { return i * nn + j; };   // expanded
    auto IDXs = [&](int i, int j) -> int { return i * n  + j; };   // source

    std::vector<double> IMGEXP(mm * nn, 0.0);
    std::vector<double> D1(mm * nn, 0.0);
    std::vector<double> D2(mm * nn, 0.0);
    std::vector<double> D3(mm * nn, 0.0);
    std::vector<double> C1(mm * nn, 0.0);
    std::vector<double> C2(mm * nn, 0.0);

    // -----------------------------------------------------------------------
    // Copy low-res grid onto high-res grid.
    // MATLAB: IMGEXP(1:2:end, 1:2:end) = IMG
    // 0-based: even rows and even cols.
    // -----------------------------------------------------------------------
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            IMGEXP[IDX(i * 2, j * 2)] = IMG[IDXs(i, j)];

    // -----------------------------------------------------------------------
    // Border interpolation (average of 2 neighbors).
    //
    // MATLAB (1-based):
    //   for i = 2:2:mm-1   <- even rows, excluding last
    //     IMGEXP(i,1)  = avg(IMGEXP(i-1,1),  IMGEXP(i+1,1))
    //     IMGEXP(i,nn) = avg(IMGEXP(i-1,nn), IMGEXP(i+1,nn))
    //
    // 0-based: odd rows from 1 to mm-2
    // -----------------------------------------------------------------------
    for (int i = 1; i <= mm - 2; i += 2) {
        IMGEXP[IDX(i, 0)]      = (IMGEXP[IDX(i-1, 0)]      + IMGEXP[IDX(i+1, 0)])      / 2.0;
        IMGEXP[IDX(i, nn-1)]   = (IMGEXP[IDX(i-1, nn-1)]   + IMGEXP[IDX(i+1, nn-1)])   / 2.0;
    }
    // MATLAB (1-based):
    //   for i = 2:2:nn      <- even cols up to nn (note: goes to nn inclusive)
    //     IMGEXP(1,i)  = avg(IMGEXP(1,i-1),  IMGEXP(1,i+1))
    //     IMGEXP(mm,i) = avg(IMGEXP(mm,i-1), IMGEXP(mm,i+1))
    //
    // 0-based: odd cols from 1 to nn-1
    // -----------------------------------------------------------------------
    for (int j = 1; j <= nn - 1; j += 2) {
        IMGEXP[IDX(0,    j)]   = (IMGEXP[IDX(0,    j-1)]   + IMGEXP[IDX(0,    j+1)])   / 2.0;
        IMGEXP[IDX(mm-1, j)]   = (IMGEXP[IDX(mm-1, j-1)]   + IMGEXP[IDX(mm-1, j+1)])   / 2.0;
    }

    // -----------------------------------------------------------------------
    // Two-phase interpolation.
    // s=0: diagonal directions.
    // s=1: vertical and horizontal directions.
    //
    // All 0-based index arithmetic below is derived from MATLAB 1-based
    // by substituting i0 = i-1, j0 = j-1 throughout.
    // -----------------------------------------------------------------------
    for (int s = 0; s <= 1; ++s) {

        // -------------------------------------------------------------------
        // FCBI pass
        //
        // MATLAB (1-based):
        //   for i = 2 : 2-s : mm-s
        //     for j = 2+(s*(1-mod(i,2))) : 2 : nn-s
        //
        // 0-based (i0=i-1, j0=j-1):
        //   i0 from 1 to mm-s-1, step 2-s
        //   j0 from (1 + s*(1 - i_1based%2)) to nn-s-1, step 2
        // -------------------------------------------------------------------
        int i_step = 2 - s;
        for (int i0 = 1; i0 <= mm - s - 1; i0 += i_step) {
            int i_1based   = i0 + 1;
            int j0_start   = 1 + s * (1 - (i_1based % 2));
            for (int j0 = j0_start; j0 <= nn - s - 1; j0 += 2) {

                // MATLAB:
                // v1 = |IMGEXP(i-1, j-1+s) - IMGEXP(i+1, j+1-s)|
                // v2 = |IMGEXP(i+1-s, j-1) - IMGEXP(i-1+s, j+1)|
                // p1 = (IMGEXP(i-1, j-1+s) + IMGEXP(i+1, j+1-s)) / 2
                // p2 = (IMGEXP(i+1-s, j-1) + IMGEXP(i-1+s, j+1)) / 2
                double A = IMGEXP[IDX(i0-1,   j0-1+s)];
                double B = IMGEXP[IDX(i0+1,   j0+1-s)];
                double C = IMGEXP[IDX(i0+1-s, j0-1)];
                double D = IMGEXP[IDX(i0-1+s, j0+1)];

                double v1 = std::abs(A - B);
                double v2 = std::abs(C - D);
                double p1 = (A + B) / 2.0;
                double p2 = (C + D) / 2.0;

                // MATLAB boundary check (1-based):
                // i > 3-s  &&  i < mm-3-s  &&  j > 3-s  &&  j < nn-3-s
                // 0-based: i0+1 > 3-s => i0 >= 3-s
                //          i0+1 < mm-3-s => i0 <= mm-5-s (i.e. i0 < mm-4-s)
                //          j0+1 > 3-s => j0 >= 3-s
                //          j0+1 < nn-3-s => j0 < nn-4-s
                bool inBounds = (i0 >= 3-s) && (i0 < mm-4-s) &&
                                (j0 >= 3-s) && (j0 < nn-4-s);

                if (v1 < tm && v2 < tm && inBounds && std::abs(p1 - p2) < tm) {
                    // MATLAB curvature comparison:
                    // lhs = |IMGEXP(i-1-s, j-3+2s) + IMGEXP(i-3+s, j-1+2s) +
                    //        IMGEXP(i+1+s, j+3-2s) + IMGEXP(i+3-s, j+1-2s) +
                    //        2*p2 - 6*p1|
                    // rhs = |IMGEXP(i-3+2s, j+1+s) + IMGEXP(i-1+2s, j+3-s) +
                    //        IMGEXP(i+3-2s, j-1-s) + IMGEXP(i+1-2s, j-3+s) +
                    //        2*p1 - 6*p2|
                    double lhs = std::abs(
                        IMGEXP[IDX(i0-1-s,   j0-3+2*s)] +
                        IMGEXP[IDX(i0-3+s,   j0-1+2*s)] +
                        IMGEXP[IDX(i0+1+s,   j0+3-2*s)] +
                        IMGEXP[IDX(i0+3-s,   j0+1-2*s)] +
                        2.0 * p2 - 6.0 * p1);
                    double rhs = std::abs(
                        IMGEXP[IDX(i0-3+2*s, j0+1+s  )] +
                        IMGEXP[IDX(i0-1+2*s, j0+3-s  )] +
                        IMGEXP[IDX(i0+3-2*s, j0-1-s  )] +
                        IMGEXP[IDX(i0+1-2*s, j0-3+s  )] +
                        2.0 * p1 - 6.0 * p2);
                    IMGEXP[IDX(i0, j0)] = (lhs > rhs) ? p1 : p2;
                } else {
                    IMGEXP[IDX(i0, j0)] = (v1 < v2) ? p1 : p2;
                }
            }
        }

        if (fcbiOnly) continue;  // FCBI only: skip iterative refinement for this s

        // -------------------------------------------------------------------
        // Iterative refinement (ICBI)
        //
        // MATLAB:
        //   step = 4.0 / (1 + s)
        //   for g = 1:ST
        //     if g < ST/4:     step = 1
        //     elseif g < ST/2: step = 2
        //     elseif g < 3*ST/4: step = 2
        //     end
        //     ...
        // -------------------------------------------------------------------
        double step = 4.0 / (1.0 + s);

        for (int g = 1; g <= st; ++g) {
            double diff = 0.0;

            // Step scheduling (mirrors MATLAB exactly)
            if (g < st / 4)
                step = 1.0;
            else if (g < st / 2)
                step = 2.0;
            else if (g < 3 * st / 4)
                step = 2.0;
            // else: keep step from previous iteration (will be 2.0)

            // ---------------------------------------------------------------
            // Compute derivatives.
            //
            // MATLAB (1-based):
            //   for i = 4-(2*s) : 1 : mm-3+s
            //     for j = 4-(2*s)+((1-s)*mod(i,2)) : 2-s : nn-3+s
            //
            // 0-based:
            //   i0 from 3-2s to mm-4+s, step 1
            //   j0 from 3-2s+(1-s)*(i_1based%2) to nn-4+s, step 2-s
            // ---------------------------------------------------------------
            for (int i0 = 3 - 2*s; i0 <= mm - 4 + s; i0 += 1) {
                int i_1based2 = i0 + 1;
                int j0_start2 = (3 - 2*s) + (1 - s) * (i_1based2 % 2);
                int j0_step   = 2 - s;
                for (int j0 = j0_start2; j0 <= nn - 4 + s; j0 += j0_step) {
                    double cur = IMGEXP[IDX(i0, j0)];

                    // MATLAB:
                    // C1 = (IMGEXP(i-1+s, j-1) - IMGEXP(i+1-s, j+1)) / 2
                    // C2 = (IMGEXP(i+1-2s, j-1+s) - IMGEXP(i-1+2s, j+1-s)) / 2
                    // D1 = IMGEXP(i-1+s,j-1) + IMGEXP(i+1-s,j+1) - 2*IMGEXP(i,j)
                    // D2 = IMGEXP(i+1,j-1+s) + IMGEXP(i-1,j+1-s) - 2*IMGEXP(i,j)
                    // D3 = (IMGEXP(i-s,j-2+s) - IMGEXP(i-2+s,j+s)
                    //      + IMGEXP(i+s,j+2-s) - IMGEXP(i+2-s,j-s)) / 2
                    double e1 = IMGEXP[IDX(i0-1+s,   j0-1)];
                    double e2 = IMGEXP[IDX(i0+1-s,   j0+1)];
                    double e3 = IMGEXP[IDX(i0+1,     j0-1+s)];
                    double e4 = IMGEXP[IDX(i0-1,     j0+1-s)];

                    C1[IDX(i0,j0)] = (e1 - e2) / 2.0;
                    C2[IDX(i0,j0)] = (IMGEXP[IDX(i0+1-2*s, j0-1+s)] -
                                       IMGEXP[IDX(i0-1+2*s, j0+1-s)]) / 2.0;
                    D1[IDX(i0,j0)] = e1 + e2 - 2.0 * cur;
                    D2[IDX(i0,j0)] = e3 + e4 - 2.0 * cur;
                    D3[IDX(i0,j0)] = (IMGEXP[IDX(i0-s,   j0-2+s)] -
                                       IMGEXP[IDX(i0-2+s, j0+s  )] +
                                       IMGEXP[IDX(i0+s,   j0+2-s)] -
                                       IMGEXP[IDX(i0+2-s, j0-s  )]) / 2.0;
                }
            }

            // ---------------------------------------------------------------
            // Energy minimization.
            //
            // MATLAB (1-based):
            //   for i = 6-(3*s) : 2-s : mm-5+(3*s)
            //     for j = 6+(s*(mod(i,2)-3)) : 2 : nn-5+(3*s)
            //
            // 0-based:
            //   i0 from 5-3s to mm-6+3s, step 2-s
            //   j0 from 5+s*(i_1based%2-3) to nn-6+3s, step 2
            // ---------------------------------------------------------------
            for (int i0 = 5 - 3*s; i0 <= mm - 6 + 3*s; i0 += (2 - s)) {
                int i_1based3  = i0 + 1;
                int j0_start3  = 5 + s * (i_1based3 % 2 - 3);
                for (int j0 = j0_start3; j0 <= nn - 6 + 3*s; j0 += 2) {

                    double cur = IMGEXP[IDX(i0, j0)];

                    // Edge continuity flags
                    // MATLAB:
                    // if |IMGEXP(i+1-s,j+1) - IMGEXP(i,j)| > TC: c_1=0 else c_1=1
                    // if |IMGEXP(i-1+s,j-1) - IMGEXP(i,j)| > TC: c_2=0 else c_2=1
                    // if |IMGEXP(i+1,j-1+s) - IMGEXP(i,j)| > TC: c_3=0 else c_3=1
                    // if |IMGEXP(i-1,j+1-s) - IMGEXP(i,j)| > TC: c_4=0 else c_4=1
                    double n_p1s = IMGEXP[IDX(i0+1-s, j0+1)];
                    double n_m1s = IMGEXP[IDX(i0-1+s, j0-1)];
                    double n_p1  = IMGEXP[IDX(i0+1,   j0-1+s)];
                    double n_m1  = IMGEXP[IDX(i0-1,   j0+1-s)];

                    int c_1 = (std::abs(n_p1s - cur) > tc) ? 0 : 1;
                    int c_2 = (std::abs(n_m1s - cur) > tc) ? 0 : 1;
                    int c_3 = (std::abs(n_p1  - cur) > tc) ? 0 : 1;
                    int c_4 = (std::abs(n_m1  - cur) > tc) ? 0 : 1;

                    double d1c = D1[IDX(i0, j0)];
                    double d2c = D2[IDX(i0, j0)];
                    double c1c = C1[IDX(i0, j0)];
                    double c2c = C2[IDX(i0, j0)];

                    double d1_p1s = D1[IDX(i0+1-s, j0+1)];
                    double d1_m1s = D1[IDX(i0-1+s, j0-1)];
                    double d1_p1  = D1[IDX(i0+1,   j0-1+s)];
                    double d1_m1  = D1[IDX(i0-1,   j0+1-s)];
                    double d2_p1s = D2[IDX(i0+1-s, j0+1)];
                    double d2_m1s = D2[IDX(i0-1+s, j0-1)];
                    double d2_p1  = D2[IDX(i0+1,   j0-1+s)];
                    double d2_m1  = D2[IDX(i0-1,   j0+1-s)];

                    // Second order terms for curvature enhancement (EN5, EN6)
                    // MATLAB:
                    // EN5 = |IMGEXP(i-2+2s,j-2) + IMGEXP(i+2-2s,j+2) - 2*IMGEXP(i,j)|
                    // EN6 = |IMGEXP(i+2,j-2+2s) + IMGEXP(i-2,j+2-2s) - 2*IMGEXP(i,j)|
                    double far1 = IMGEXP[IDX(i0-2+2*s, j0-2    )] +
                                  IMGEXP[IDX(i0+2-2*s, j0+2    )] - 2.0*cur;
                    double far2 = IMGEXP[IDX(i0+2,     j0-2+2*s)] +
                                  IMGEXP[IDX(i0-2,     j0+2-2*s)] - 2.0*cur;

                    // Current energy terms
                    double EN1 = c_1*std::abs(d1c-d1_p1s) + c_2*std::abs(d1c-d1_m1s);
                    double EN2 = c_3*std::abs(d1c-d1_p1 ) + c_4*std::abs(d1c-d1_m1 );
                    double EN3 = c_1*std::abs(d2c-d2_p1s) + c_2*std::abs(d2c-d2_m1s);
                    double EN4 = c_3*std::abs(d2c-d2_p1 ) + c_4*std::abs(d2c-d2_m1 );
                    double EN5 = std::abs(far1);
                    double EN6 = std::abs(far2);

                    // +step energy terms
                    double EA1 = c_1*std::abs(d1c-d1_p1s-3.0*step) + c_2*std::abs(d1c-d1_m1s-3.0*step);
                    double EA2 = c_3*std::abs(d1c-d1_p1 -3.0*step) + c_4*std::abs(d1c-d1_m1 -3.0*step);
                    double EA3 = c_1*std::abs(d2c-d2_p1s-3.0*step) + c_2*std::abs(d2c-d2_m1s-3.0*step);
                    double EA4 = c_3*std::abs(d2c-d2_p1 -3.0*step) + c_4*std::abs(d2c-d2_m1 -3.0*step);
                    double EA5 = std::abs(far1 - 2.0*step);
                    double EA6 = std::abs(far2 - 2.0*step);

                    // -step energy terms
                    double ES1 = c_1*std::abs(d1c-d1_p1s+3.0*step) + c_2*std::abs(d1c-d1_m1s+3.0*step);
                    double ES2 = c_3*std::abs(d1c-d1_p1 +3.0*step) + c_4*std::abs(d1c-d1_m1 +3.0*step);
                    double ES3 = c_1*std::abs(d2c-d2_p1s+3.0*step) + c_2*std::abs(d2c-d2_m1s+3.0*step);
                    double ES4 = c_3*std::abs(d2c-d2_p1 +3.0*step) + c_4*std::abs(d2c-d2_m1 +3.0*step);
                    double ES5 = std::abs(far1 + 2.0*step);
                    double ES6 = std::abs(far2 + 2.0*step);

                    // Isophote curvature
                    // MATLAB:
                    // EISO = (C1^2*D2 - 2*C1*C2*D3 + C2^2*D1) / (C1^2 + C2^2)
                    // if |EISO| < 0.2: EISO = 0
                    double denom_iso = c1c*c1c + c2c*c2c;
                    double EISO = 0.0;
                    if (denom_iso > 1e-10) {
                        EISO = (c1c*c1c*d2c
                              - 2.0*c1c*c2c*D3[IDX(i0,j0)]
                              + c2c*c2c*d1c) / denom_iso;
                    }
                    if (std::abs(EISO) < 0.2) EISO = 0.0;
                    double sgn = (EISO > 0.0) ? 1.0 : (EISO < 0.0 ? -1.0 : 0.0);

                    // Combined energy (PF selects which terms are active)
                    double EN, EA, ES;
                    if (pf == 1) {
                        EN = al*(EN1+EN2+EN3+EN4) + bt*(EN5+EN6);
                        EA = al*(EA1+EA2+EA3+EA4) + bt*(EA5+EA6);
                        ES = al*(ES1+ES2+ES3+ES4) + bt*(ES5+ES6);
                    } else if (pf == 2) {
                        EN = al*(EN1+EN2+EN3+EN4);
                        EA = al*(EA1+EA2+EA3+EA4) - gm*sgn;
                        ES = al*(ES1+ES2+ES3+ES4) + gm*sgn;
                    } else {
                        EN = al*(EN1+EN2+EN3+EN4) + bt*(EN5+EN6);
                        EA = al*(EA1+EA2+EA3+EA4) + bt*(EA5+EA6) - gm*sgn;
                        ES = al*(ES1+ES2+ES3+ES4) + bt*(ES5+ES6) + gm*sgn;
                    }

                    // MATLAB:
                    // if EN>EA && ES>EA: pixel += step
                    // elseif EN>ES && EA>ES: pixel -= step
                    if (EN > EA && ES > EA) {
                        IMGEXP[IDX(i0, j0)] += step;
                        diff += step;
                    } else if (EN > ES && EA > ES) {
                        IMGEXP[IDX(i0, j0)] -= step;
                        diff += step;
                    }
                }
            }

            // MATLAB: if SC==1 && diff < TS: break
            if (sc == 1 && diff < ts) break;

        } // end iterative refinement (g loop)

    } // end s = 0, 1

    return IMGEXP;
}

// -------------------------------------------------------------------------
// scaleICBI - public entry point
//
// ZK selection: find the number of ICBI doubling passes that produces an
// output size CLOSEST to the requested output size (outW x outH).
// The "closest" criterion minimises max(|icbi_w/outW - 1|, |icbi_h/outH - 1|).
// This means:
//   --scale 2.0 on 100px -> ICBI 1-pass gives 199, deviation=0.5%  <- chosen
//                           ICBI 2-pass gives 397, deviation=98.5%
//   --scale 4.0 on 100px -> ICBI 1-pass gives 199, deviation=50%
//                           ICBI 2-pass gives 397, deviation=0.75% <- chosen
//   --scale 2.5 on 100px -> ICBI 1-pass gives 199, deviation=20%   <- chosen
//                           ICBI 2-pass gives 397, deviation=59%
//
// After ICBI doubling(s), a bilinear resample brings the result to exactly
// (outW x outH). For near-exact cases like scale=2.0 this is a <1% resample.
// -------------------------------------------------------------------------
void scaleICBI(const unsigned char* input, int inW, int inH,
               unsigned char* output, int outW, int outH,
               int sz, int pf, bool vr, int st,
               double tm, double tc, int sc, double ts,
               double al, double bt, double gm,
               bool fcbiOnly)
{
    // -----------------------------------------------------------------------
    // Determine optimal number of ICBI doubling passes (ZK)
    // -----------------------------------------------------------------------
    int bestZK  = 1;
    double bestDev = 1e18;

    {
        int tw = inW, th = inH;
        for (int zk = 1; zk <= 8; ++zk) {
            tw = 2*tw - 1;
            th = 2*th - 1;
            // Deviation: how far is this ICBI output from the target?
            // Use max of width and height relative deviations.
            double devW = std::abs((double)tw / outW - 1.0);
            double devH = std::abs((double)th / outH - 1.0);
            double dev  = std::max(devW, devH);
            if (dev < bestDev) {
                bestDev = dev;
                bestZK  = zk;
            }
            // Early exit: once both dimensions are more than 3x the target,
            // further passes will only get worse.
            if (tw > outW * 3 && th > outH * 3) break;
        }
    }

    if (vr) {
        // Compute what the chosen ZK gives
        int tw = inW, th = inH;
        for (int zk = 0; zk < bestZK; ++zk) { tw = 2*tw-1; th = 2*th-1; }
        std::cout << "  [ICBI] ZK=" << bestZK
                  << "  ICBI native output: " << tw << "x" << th
                  << "  -> final resample to: " << outW << "x" << outH << std::endl;
    }

    const int CL = 3;  // process RGB channels; alpha handled separately

    // -----------------------------------------------------------------------
    // Convert input to per-channel double buffers
    // -----------------------------------------------------------------------
    int curW = inW, curH = inH;

    std::vector<std::vector<double>> channels(CL);
    for (int c = 0; c < CL; ++c) {
        channels[c].resize(inW * inH);
        for (int i = 0; i < inW * inH; ++i)
            channels[c][i] = static_cast<double>(input[i * 4 + c]);
    }

    std::vector<double> alphaChannel(inW * inH);
    for (int i = 0; i < inW * inH; ++i)
        alphaChannel[i] = static_cast<double>(input[i * 4 + 3]);

    // -----------------------------------------------------------------------
    // Apply bestZK doubling passes
    // -----------------------------------------------------------------------
    for (int zf = 0; zf < bestZK; ++zf) {
        int newW = 2*curW - 1;
        int newH = 2*curH - 1;

        if (vr) std::cout << "  [ICBI] Pass " << zf+1 << "/" << bestZK
                          << "  " << curW << "x" << curH
                          << " -> " << newW << "x" << newH << std::endl;

        // ICBI doubling for each RGB channel
        // icbiDoubleChannel(IMG, m=rows, n=cols, ...)
        for (int c = 0; c < CL; ++c) {
            channels[c] = icbiDoubleChannel(channels[c], curH, curW,
                                             pf, vr, st, tm, tc, sc, ts,
                                             al, bt, gm, fcbiOnly);
        }

        // Alpha channel: simple bilinear doubling
        {
            std::vector<double> newAlpha(newW * newH, 0.0);

            // Copy known (even-index) samples
            for (int i = 0; i < curH; ++i)
                for (int j = 0; j < curW; ++j)
                    newAlpha[(i*2)*newW + (j*2)] = alphaChannel[i*curW + j];

            // Left/right border vertical midpoints
            for (int i = 1; i <= newH - 2; i += 2) {
                newAlpha[i*newW + 0]        = (newAlpha[(i-1)*newW+0]      + newAlpha[(i+1)*newW+0])      / 2.0;
                newAlpha[i*newW + newW-1]   = (newAlpha[(i-1)*newW+newW-1] + newAlpha[(i+1)*newW+newW-1]) / 2.0;
            }
            // Top/bottom border horizontal midpoints
            for (int j = 1; j <= newW - 1; j += 2) {
                newAlpha[0*newW + j]        = (newAlpha[0*newW+j-1]        + newAlpha[0*newW+j+1])        / 2.0;
                newAlpha[(newH-1)*newW + j] = (newAlpha[(newH-1)*newW+j-1] + newAlpha[(newH-1)*newW+j+1]) / 2.0;
            }
            // Interior points
            for (int i = 0; i < newH; ++i) {
                for (int j = 0; j < newW; ++j) {
                    if (i % 2 == 0 && j % 2 == 0) continue;  // already set
                    if (i % 2 == 1 && j % 2 == 1) {
                        // Diagonal midpoint: average of 4 diagonal neighbours
                        double v = 0; int cnt = 0;
                        if (i>0       && j>0      ) { v += newAlpha[(i-1)*newW+(j-1)]; cnt++; }
                        if (i>0       && j<newW-1 ) { v += newAlpha[(i-1)*newW+(j+1)]; cnt++; }
                        if (i<newH-1  && j>0      ) { v += newAlpha[(i+1)*newW+(j-1)]; cnt++; }
                        if (i<newH-1  && j<newW-1 ) { v += newAlpha[(i+1)*newW+(j+1)]; cnt++; }
                        if (cnt) newAlpha[i*newW+j] = v / cnt;
                    } else if (i % 2 == 1) {
                        // Vertical midpoint
                        if (i > 0 && i < newH-1)
                            newAlpha[i*newW+j] = (newAlpha[(i-1)*newW+j] + newAlpha[(i+1)*newW+j]) / 2.0;
                    } else {
                        // Horizontal midpoint
                        if (j > 0 && j < newW-1)
                            newAlpha[i*newW+j] = (newAlpha[i*newW+j-1] + newAlpha[i*newW+j+1]) / 2.0;
                    }
                }
            }
            alphaChannel = std::move(newAlpha);
        }

        curW = newW;
        curH = newH;
    }

    // -----------------------------------------------------------------------
    // Clamp, round, and write to output.
    // If curW/curH matches outW/outH exactly, write directly.
    // Otherwise bilinear-resample to exact requested size.
    // -----------------------------------------------------------------------
    double maxVal = static_cast<double>((1 << sz) - 1);

    if (curW == outW && curH == outH) {
        // Exact match — no resampling needed
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < outW; ++j) {
                int idx = i * outW + j;
                for (int c = 0; c < CL; ++c) {
                    double v = std::round(channels[c][idx]);
                    output[idx*4+c] = static_cast<unsigned char>(
                        std::max(0.0, std::min(maxVal, v)));
                }
                double a = std::round(alphaChannel[idx]);
                output[idx*4+3] = static_cast<unsigned char>(
                    std::max(0.0, std::min(255.0, a)));
            }
        }
    } else {
        // Bilinear resample from (curW x curH) to (outW x outH)
        if (vr) std::cout << "  [ICBI] Bilinear resample: "
                          << curW << "x" << curH
                          << " -> " << outW << "x" << outH << std::endl;

        // Build intermediate RGBA byte image
        std::vector<unsigned char> tmp((size_t)curW * curH * 4);
        for (int i = 0; i < curH; ++i) {
            for (int j = 0; j < curW; ++j) {
                int idx = i * curW + j;
                for (int c = 0; c < CL; ++c) {
                    double v = std::round(channels[c][idx]);
                    tmp[idx*4+c] = static_cast<unsigned char>(
                        std::max(0.0, std::min(maxVal, v)));
                }
                double a = std::round(alphaChannel[idx]);
                tmp[idx*4+3] = static_cast<unsigned char>(
                    std::max(0.0, std::min(255.0, a)));
            }
        }

        // Bilinear resample
        double sx = static_cast<double>(curW) / outW;
        double sy = static_cast<double>(curH) / outH;
        for (int y = 0; y < outH; ++y) {
            double fy = (y + 0.5) * sy - 0.5;
            int y0 = static_cast<int>(std::floor(fy));
            int y1 = y0 + 1;
            double dy = fy - y0;
            y0 = std::max(0, std::min(curH-1, y0));
            y1 = std::max(0, std::min(curH-1, y1));
            for (int x = 0; x < outW; ++x) {
                double fx = (x + 0.5) * sx - 0.5;
                int x0 = static_cast<int>(std::floor(fx));
                int x1 = x0 + 1;
                double dx = fx - x0;
                x0 = std::max(0, std::min(curW-1, x0));
                x1 = std::max(0, std::min(curW-1, x1));
                int dst = (y * outW + x) * 4;
                for (int c = 0; c < 4; ++c) {
                    double v = (1-dx)*(1-dy) * tmp[(y0*curW+x0)*4+c]
                             +    dx *(1-dy) * tmp[(y0*curW+x1)*4+c]
                             + (1-dx)*   dy  * tmp[(y1*curW+x0)*4+c]
                             +    dx *   dy  * tmp[(y1*curW+x1)*4+c];
                    output[dst+c] = static_cast<unsigned char>(
                        std::max(0.0, std::min(255.0, std::round(v))));
                }
            }
        }
    }
}
