// =============================================================================
// fsr3_main.cpp
// FSR 3.1.5 CPU Port — Static image upscaling entry point
//
// DATA FLOW (identical design to fsr2_main.cpp):
//   Input PNG at render resolution → N jitter frames → FSR3.1.5 passes
//   → display resolution output.
//
// FSR3 uses --algo fsr3 in the CLI.
// Jitter, frames, modes, RCAS, LFGA, TEPD all work the same as FSR2.
//
// BEST QUALITY COMMANDS (same findings as FSR2):
//   image-resizer.exe in.png out.png --algo fsr3 --fsr3-jitter off --fsr3-frames 1
//   image-resizer.exe in.png out.png --algo fsr3 --onlyenablepasses3 4,6
//   image-resizer.exe in.png out.png --algo fsr3  (default, all passes, jitter on)
// =============================================================================
#include "fsr3_context.h"
#include "fsr2_jitter.h"
#include "fsr2_jitter_resample.h"
#include "fsr_math.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <set>
#include <iostream>
#include <algorithm>

static void uint8ToFloat(const unsigned char* src, float* dst, int pixels) {
    for (int i = 0; i < pixels * 4; i++) dst[i] = src[i] / 255.0f;
}
static void floatToUint8(const float* src, unsigned char* dst, int pixels) {
    for (int i = 0; i < pixels * 4; i++)
        dst[i] = (unsigned char)(clamp(src[i], 0.0f, 1.0f) * 255.0f + 0.5f);
}

void scaleFSR3(
    const unsigned char* input,    int inW,  int inH,
    unsigned char*       output,   int outW, int outH,
    float  sharpness,
    bool   useRcas,
    bool   rcasDenoise,
    float  lfga,
    bool   useTepd,
    float  depth,
    bool   useJitter,
    int    numFrames,
    Fsr2JitterMode       jitterMode,
    const std::set<int>& enabledPasses)
{
    size_t renderPixels  = (size_t)inW  * inH;
    size_t displayPixels = (size_t)outW * outH;

    std::vector<float> inputF(renderPixels * 4);
    uint8ToFloat(input, inputF.data(), (int)renderPixels);

    int phaseCount   = useJitter ? fsr2GetJitterPhaseCount(inW, outW) : 1;
    int actualFrames = std::max(1, numFrames);

    std::cout << "  [FSR3] Render: " << inW << "x" << inH
              << " -> Display: " << outW << "x" << outH << std::endl;
    std::cout << "  [FSR3] Jitter: " << (useJitter ? "ON" : "OFF")
              << "  Frames: " << actualFrames
              << "  PhaseCount: " << phaseCount << std::endl;
    std::cout << "  [FSR3] RCAS: " << (useRcas ? "ON" : "OFF");
    if (useRcas)
        std::cout << "  Sharpness: " << sharpness
                  << " (exp2(-" << sharpness << ")=" << std::exp2(-sharpness) << ")";
    std::cout << std::endl;
    std::cout << "  [FSR3] Enabled passes: ";
    for (int p : enabledPasses) std::cout << p << " ";
    std::cout << std::endl;

    std::vector<float> depthBuf(renderPixels, depth);
    std::vector<float> mvBuf(renderPixels * 2, 0.0f);

    Fsr3Context ctx;
    ctx.init(inW, inH, outW, outH);

    std::vector<float> displayF(displayPixels * 4, 0.0f);
    std::vector<float> jitteredFrame(renderPixels * 4);

    for (int fi = 0; fi < actualFrames; fi++) {
        float jX = 0.0f, jY = 0.0f;
        if (useJitter)
            fsr2GetJitterOffset(&jX, &jY, fi, phaseCount);

        if (useJitter && (std::abs(jX) > 1e-7f || std::abs(jY) > 1e-7f)) {
            fsr2ResampleWithShift(inputF.data(), inW, inH,
                                   jX, jY,
                                   jitteredFrame.data(), jitterMode);
        } else {
            std::copy(inputF.begin(), inputF.end(), jitteredFrame.begin());
        }

        Fsr3DispatchParams p;
        p.color         = jitteredFrame.data();
        p.depth         = depthBuf.data();
        p.motionVectors = mvBuf.data();
        p.reactive      = nullptr;
        p.transparencyAndComposition = nullptr;
        p.renderW       = inW;  p.renderH  = inH;
        p.displayW      = outW; p.displayH = outH;
        p.jitterOffsetX = jX;   p.jitterOffsetY = jY;
        p.motionVectorScaleX = 1.0f;
        p.motionVectorScaleY = 1.0f;
        p.frameTimeDelta     = 16.667f;
        p.sharpness          = sharpness;
        p.enableSharpening   = enabledPasses.count(FSR3_PASS_ACCUMULATE_SHARPEN) > 0;
        p.flags              = 0;
        p.reset              = (fi == 0);

        fsr3Dispatch(ctx, p, enabledPasses, displayF.data(), rcasDenoise);

        if (actualFrames > 8 && (fi + 1) % 8 == 0) {
            std::cout << "  [FSR3] Accumulated " << (fi+1)
                      << "/" << actualFrames << " frames..." << std::endl;
        }
    }

    // Apply LFGA/TEPD post-processing
    for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++) {
            size_t idx = ((size_t)y * outW + x) * 4;
            float3 color(displayF[idx], displayF[idx+1], displayF[idx+2]);
            color = applyPostProcess(color, x, y, lfga, useTepd);
            displayF[idx+0] = color.x;
            displayF[idx+1] = color.y;
            displayF[idx+2] = color.z;
        }
    }

    floatToUint8(displayF.data(), output, (int)displayPixels);
}
