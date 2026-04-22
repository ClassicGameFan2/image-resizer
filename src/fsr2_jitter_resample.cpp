// =============================================================================
// fsr2_jitter_resample.cpp
// FSR 2.3.4 CPU Port — Sub-pixel jitter frame generator implementation
// =============================================================================
#include "fsr2_jitter_resample.h"
#include <cmath>
#include <algorithm>

static inline float clampF(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline void fetchPixelF(const float* src, int w, int h, int px, int py, float out[4]) {
    px = std::max(0, std::min(w - 1, px));
    py = std::max(0, std::min(h - 1, py));
    const float* p = src + ((size_t)py * w + px) * 4;
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; out[3] = p[3];
}

// ── Bilinear ─────────────────────────────────────────────────────────────────
static void jitterBilinear(const float* src, int w, int h, float shiftX, float shiftY, float* dst) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sx = (float)x - shiftX;
            float sy = (float)y - shiftY;
            int x0 = (int)std::floor(sx);
            int y0 = (int)std::floor(sy);
            float fx = sx - (float)x0;
            float fy = sy - (float)y0;

            float p00[4], p10[4], p01[4], p11[4];
            fetchPixelF(src, w, h, x0,   y0,   p00);
            fetchPixelF(src, w, h, x0+1, y0,   p10);
            fetchPixelF(src, w, h, x0,   y0+1, p01);
            fetchPixelF(src, w, h, x0+1, y0+1, p11);

            float* d = dst + ((size_t)y * w + x) * 4;
            for (int c = 0; c < 4; c++) {
                float top    = p00[c] + fx * (p10[c] - p00[c]);
                float bottom = p01[c] + fx * (p11[c] - p01[c]);
                d[c] = top + fy * (bottom - top);
            }
        }
    }
}

// ── Bicubic (Catmull-Rom) ─────────────────────────────────────────────────────
static float cubicW(float t) {
    t = std::abs(t);
    if (t < 1.0f) return 1.5f*t*t*t - 2.5f*t*t + 1.0f;
    if (t < 2.0f) return -0.5f*t*t*t + 2.5f*t*t - 4.0f*t + 2.0f;
    return 0.0f;
}

static void jitterBicubic(const float* src, int w, int h, float shiftX, float shiftY, float* dst) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sx = (float)x - shiftX;
            float sy = (float)y - shiftY;
            int x0 = (int)std::floor(sx);
            int y0 = (int)std::floor(sy);
            float fx = sx - (float)x0;
            float fy = sy - (float)y0;

            float result[4] = {0,0,0,0};
            for (int ky = -1; ky <= 2; ky++) {
                float wy = cubicW((float)ky - fy);
                for (int kx = -1; kx <= 2; kx++) {
                    float wx = cubicW((float)kx - fx);
                    float p[4];
                    fetchPixelF(src, w, h, x0+kx, y0+ky, p);
                    for (int c = 0; c < 4; c++)
                        result[c] += wx * wy * p[c];
                }
            }

            float* d = dst + ((size_t)y * w + x) * 4;
            for (int c = 0; c < 4; c++)
                d[c] = clampF(result[c], 0.0f, 1.0f);
        }
    }
}

// ── Lanczos-3 ─────────────────────────────────────────────────────────────────
static const float kPi = 3.14159265358979323846f;
static float jSinc(float x) {
    if (std::abs(x) < 1e-6f) return 1.0f;
    float px = kPi * x;
    return std::sin(px) / px;
}
static float lanczos3W(float x) {
    x = std::abs(x);
    if (x >= 3.0f) return 0.0f;
    return jSinc(x) * jSinc(x / 3.0f);
}

static void jitterLanczos3(const float* src, int w, int h, float shiftX, float shiftY, float* dst) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sx = (float)x - shiftX;
            float sy = (float)y - shiftY;
            int x0 = (int)std::floor(sx);
            int y0 = (int)std::floor(sy);
            float fx = sx - (float)x0;
            float fy = sy - (float)y0;

            float result[4] = {0,0,0,0};
            float wsum = 0.0f;
            for (int ky = -2; ky <= 3; ky++) {
                float wy = lanczos3W((float)ky - fy);
                for (int kx = -2; kx <= 3; kx++) {
                    float wx = lanczos3W((float)kx - fx);
                    float w2 = wx * wy;
                    float p[4];
                    fetchPixelF(src, w, h, x0+kx, y0+ky, p);
                    for (int c = 0; c < 4; c++)
                        result[c] += w2 * p[c];
                    wsum += w2;
                }
            }

            float* d = dst + ((size_t)y * w + x) * 4;
            if (wsum > 1e-6f) {
                for (int c = 0; c < 4; c++)
                    d[c] = clampF(result[c] / wsum, 0.0f, 1.0f);
            } else {
                float p[4]; fetchPixelF(src, w, h, x0, y0, p);
                for (int c = 0; c < 4; c++) d[c] = p[c];
            }
        }
    }
}

// ── Public ────────────────────────────────────────────────────────────────────
void fsr2ResampleWithShift(const float* src, int srcW, int srcH,
                            float shiftX, float shiftY,
                            float* dst, Fsr2JitterMode mode) {
    switch (mode) {
    case Fsr2JitterMode::BILINEAR:  jitterBilinear(src, srcW, srcH, shiftX, shiftY, dst); break;
    case Fsr2JitterMode::BICUBIC:   jitterBicubic(src, srcW, srcH, shiftX, shiftY, dst); break;
    case Fsr2JitterMode::LANCZOS3:  jitterLanczos3(src, srcW, srcH, shiftX, shiftY, dst); break;
    }
}
