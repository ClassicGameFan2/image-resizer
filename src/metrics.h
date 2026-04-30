#pragma once
#include <string>
#include <vector>

struct MetricResults {
    double psnr = -1.0;
    double ssim = -1.0;
    double fsim = -1.0;
    bool computePsnr = false;
    bool computeSsim = false;
    bool computeFsim = false;
};

// Compute requested metrics between two RGBA images of the same size.
// img1 = reference, img2 = distorted
MetricResults computeMetrics(
    const unsigned char* img1, const unsigned char* img2,
    int width, int height,
    bool doPsnr, bool doSsim, bool doFsim);

void printMetrics(const MetricResults& r);
