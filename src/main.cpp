#include <iostream>
#include <string>
#include <vector>
#include "stb_image.h"
#include "stb_image_write.h"

extern void scaleFSR_EASU(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH);
extern void applyFSR_RCAS(const unsigned char* input, int w, int h, unsigned char* output, float sharpness);

// We bring back Nearest Neighbor!
void scaleNearestNeighbor(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH) {
    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            int srcX = (x * inW) / outW;
            int srcY = (y * inH) / outH;
            int srcIdx = (srcY * inW + srcX) * 4;
            int dstIdx = (y * outW + x) * 4;
            for(int i=0; i<4; i++) output[dstIdx + i] = input[srcIdx + i];
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "--- Image Resizer CPU ---" << std::endl;

    std::string inputFile = "";
    std::string outputFile = "";
    float scale = 2.0f;
    float sharpness = 0.2f;
    std::string algo = "fsr"; // Default to FSR

    // Parse command line flags
    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--scale" && i + 1 < argc) scale = std::stof(argv[++i]);
        else if (arg == "--algo" && i + 1 < argc) algo = argv[++i];
        else if (arg == "--sharpness" && i + 1 < argc) sharpness = std::stof(argv[++i]);
        else if (inputFile.empty()) inputFile = arg;
        else if (outputFile.empty()) outputFile = arg;
    }

    if (inputFile.empty() || outputFile.empty()) {
        std::cout << "Usage: image-resizer.exe <in.png> <out.png> [--scale 2.0] [--algo fsr|nearest] [--sharpness 0.2]" << std::endl;
        return 1;
    }

    int width, height, channels;
    unsigned char* imgData = stbi_load(inputFile.c_str(), &width, &height, &channels, 4);
    if (!imgData) {
        std::cout << "Failed to load image!" << std::endl;
        return 1;
    }

    int newW = (int)(width * scale);
    int newH = (int)(height * scale);
    unsigned char* finalData = new unsigned char[newW * newH * 4];

    if (algo == "nearest") {
        std::cout << "Processing (Nearest Neighbor)..." << std::endl;
        scaleNearestNeighbor(imgData, width, height, finalData, newW, newH);
    } 
    else if (algo == "fsr") {
        std::cout << "Processing (AMD FSR 1.0)..." << std::endl;
        unsigned char* easuData = new unsigned char[newW * newH * 4];
        scaleFSR_EASU(imgData, width, height, easuData, newW, newH);
        applyFSR_RCAS(easuData, newW, newH, finalData, sharpness);
        delete[] easuData;
    } 
    else {
        std::cout << "Unknown algorithm: " << algo << std::endl;
        return 1;
    }

    std::cout << "Saving: " << outputFile << std::endl;
    stbi_write_png(outputFile.c_str(), newW, newH, 4, finalData, newW * 4);

    stbi_image_free(imgData);
    delete[] finalData;
    return 0;
}
