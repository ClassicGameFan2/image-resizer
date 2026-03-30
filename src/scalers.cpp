#include <cmath>
#include <algorithm>

// --------------------------------------------------------------------------
// MATH CURVES FOR SCALING
// --------------------------------------------------------------------------

double weight_bilinear(double x) {
    x = std::abs(x);
    if (x < 1.0) return 1.0 - x;
    return 0.0;
}

double weight_bicubic(double x) {
    // Catmull-Rom Spline (Crisp Bicubic)
    x = std::abs(x);
    double a = -0.5; 
    if (x <= 1.0) return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
    if (x < 2.0)  return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
    return 0.0;
}

double sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= 3.14159265358979323846;
    return std::sin(x) / x;
}

double weight_lanczos3(double x) {
    x = std::abs(x);
    if (x < 3.0) return sinc(x) * sinc(x / 3.0);
    return 0.0;
}

unsigned char clamp_byte(double v) {
    return (unsigned char)std::max(0.0, std::min(255.0, v + 0.5)); // +0.5 for rounding
}

// --------------------------------------------------------------------------
// GENERIC RESAMPLER (Handles both upscaling and perfect downscaling!)
// --------------------------------------------------------------------------
typedef double (*KernelFunc)(double);

void processGenericResample(const unsigned char* input, int inW, int inH, 
                            unsigned char* output, int outW, int outH, 
                            KernelFunc kernel, double baseRadius) {
    
    double scaleX = (double)outW / (double)inW;
    double scaleY = (double)outH / (double)inH;
    
    // If downscaling, we expand the filter radius to act as anti-aliasing
    double filterScaleX = std::min(1.0, scaleX);
    double filterScaleY = std::min(1.0, scaleY);
    
    double radiusX = baseRadius / filterScaleX;
    double radiusY = baseRadius / filterScaleY;

    for (int y = 0; y < outH; ++y) {
        double srcY = (y + 0.5) / scaleY - 0.5;
        int yMin = std::max(0, (int)std::floor(srcY - radiusY + 1));
        int yMax = std::min(inH - 1, (int)std::floor(srcY + radiusY));

        for (int x = 0; x < outW; ++x) {
            double srcX = (x + 0.5) / scaleX - 0.5;
            int xMin = std::max(0, (int)std::floor(srcX - radiusX + 1));
            int xMax = std::min(inW - 1, (int)std::floor(srcX + radiusX));

            double r = 0, g = 0, b = 0, a = 0;
            double weightSum = 0.0;

            for (int cy = yMin; cy <= yMax; ++cy) {
                double dy = (srcY - cy) * filterScaleY;
                double wy = kernel(dy);
                
                for (int cx = xMin; cx <= xMax; ++cx) {
                    double dx = (srcX - cx) * filterScaleX;
                    double wx = kernel(dx);
                    
                    double w = wx * wy;
                    int srcIdx = (cy * inW + cx) * 4;
                    r += input[srcIdx + 0] * w;
                    g += input[srcIdx + 1] * w;
                    b += input[srcIdx + 2] * w;
                    a += input[srcIdx + 3] * w;
                    weightSum += w;
                }
            }
            
            int dstIdx = (y * outW + x) * 4;
            if (weightSum > 0.0) {
                output[dstIdx + 0] = clamp_byte(r / weightSum);
                output[dstIdx + 1] = clamp_byte(g / weightSum);
                output[dstIdx + 2] = clamp_byte(b / weightSum);
                output[dstIdx + 3] = clamp_byte(a / weightSum);
            }
        }
    }
}

// --------------------------------------------------------------------------
// ALGORITHM WRAPPERS
// --------------------------------------------------------------------------

void scaleNearestNeighbor(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH) {
    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            int srcX = std::min(inW - 1, (x * inW) / outW);
            int srcY = std::min(inH - 1, (y * inH) / outH);
            int srcIdx = (srcY * inW + srcX) * 4;
            int dstIdx = (y * outW + x) * 4;
            for(int i=0; i<4; i++) output[dstIdx + i] = input[srcIdx + i];
        }
    }
}

void scaleBilinear(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH) {
    processGenericResample(input, inW, inH, output, outW, outH, weight_bilinear, 1.0);
}

void scaleBicubic(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH) {
    processGenericResample(input, inW, inH, output, outW, outH, weight_bicubic, 2.0);
}

void scaleLanczos3(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH) {
    processGenericResample(input, inW, inH, output, outW, outH, weight_lanczos3, 3.0);
}
