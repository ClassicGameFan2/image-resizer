#pragma once

void scaleICBI(const unsigned char* input, int inW, int inH,
               unsigned char* output, int outW, int outH,
               int   sz,   // bits per channel (default 8)
               int   pf,   // potential function 1,2,3 (default 1)
               bool  vr,   // verbose (default false)
               int   st,   // max iterations (default 20)
               double tm,  // max edge step (default 100)
               double tc,  // edge continuity threshold (default 50)
               int   sc,   // stopping criterion 1=threshold,0=fixed (default 1)
               double ts,  // threshold on change (default 100)
               double al,  // curvature continuity weight (default 1.0)
               double bt,  // curvature enhancement weight (default -1.0)
               double gm,  // isophote smoothing weight (default 5.0)
               bool fcbiOnly // if true, skip iterative refinement
               );
