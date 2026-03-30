#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

extern void scaleFSR_EASU(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH, float lfga, bool tepd);
extern void applyFSR_RCAS(const unsigned char* input, int w, int h, unsigned char* output, float sharpness, bool useDenoise, float lfga, bool tepd);

extern void scaleNearestNeighbor(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH);
extern void scaleBilinear(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH, float lfga, bool tepd);
extern void scaleBicubic(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH, float lfga, bool tepd);
extern void scaleLanczos3(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH, float lfga, bool tepd);

bool processImage(const std::string& inFile, const std::string& outFile, float scale, const std::string& algo, 
                  bool useRcas, float sharpness, bool rcasDenoise, float lfga, bool tepd, int bpp) {
                  
    int width, height, channels;
    unsigned char* imgData = stbi_load(inFile.c_str(), &width, &height, &channels, 4);
    if (!imgData) return false;

    int newW = width;
    int newH = height;
    if (algo != "off") {
        newW = std::max(1, (int)(width * scale));
        newH = std::max(1, (int)(height * scale));
    }

    unsigned char* step1Data = new unsigned char[newW * newH * 4];

    // Note: If RCAS is ON, we pass Post-Processing logic to RCAS instead!
    float pass1_lfga = useRcas ? 0.0f : lfga;
    bool pass1_tepd = useRcas ? false : tepd;

    if (algo == "off") std::memcpy(step1Data, imgData, newW * newH * 4);
    else if (algo == "nearest") scaleNearestNeighbor(imgData, width, height, step1Data, newW, newH);
    else if (algo == "bilinear") scaleBilinear(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
    else if (algo == "bicubic") scaleBicubic(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
    else if (algo == "lanczos3") scaleLanczos3(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
    else if (algo == "fsr") {
        if (scale < 1.0f) {
            std::cout << "Error: FSR mathematically cannot downscale." << std::endl;
            stbi_image_free(imgData); delete[] step1Data; return false;
        }
        scaleFSR_EASU(imgData, width, height, step1Data, newW, newH, pass1_lfga, pass1_tepd);
    } 

    unsigned char* finalData = step1Data;
    if (useRcas) {
        finalData = new unsigned char[newW * newH * 4];
        applyFSR_RCAS(step1Data, newW, newH, finalData, sharpness, rcasDenoise, lfga, tepd);
        delete[] step1Data;
    }

    // 24-BIT vs 32-BIT PACKING
    int outChannels = 4;
    unsigned char* saveBuffer = finalData;
    
    if (bpp == 24) {
        outChannels = 3;
        saveBuffer = new unsigned char[newW * newH * 3];
        for (int i = 0; i < newW * newH; ++i) {
            saveBuffer[i * 3 + 0] = finalData[i * 4 + 0];
            saveBuffer[i * 3 + 1] = finalData[i * 4 + 1];
            saveBuffer[i * 3 + 2] = finalData[i * 4 + 2];
        }
    }

    stbi_write_png(outFile.c_str(), newW, newH, outChannels, saveBuffer, newW * outChannels);

    stbi_image_free(imgData);
    delete[] finalData;
    if (bpp == 24) delete[] saveBuffer;
    return true;
}

int main(int argc, char** argv) {
    std::cout << "--- Image Resizer CPU ---" << std::endl;

    std::string inputPath = "", outputPath = "", suffix = "", algo = "fsr", rcasInput = "auto";
    float scale = 2.0f, sharpness = 0.2f, lfga = 0.0f;
    bool rcasDenoise = false, tepd = false;
    int bpp = 32;

    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--scale" && i + 1 < argc) scale = std::stof(argv[++i]);
        else if (arg == "--algo" && i + 1 < argc) algo = argv[++i];
        else if (arg == "--sharpness" && i + 1 < argc) sharpness = std::stof(argv[++i]);
        else if (arg == "--suffix" && i + 1 < argc) suffix = argv[++i];
        else if (arg == "--rcas" && i + 1 < argc) rcasInput = argv[++i];
        else if (arg == "--rcas-denoise" && i + 1 < argc) rcasDenoise = (std::string(argv[++i]) == "on");
        else if (arg == "--lfga" && i + 1 < argc) lfga = std::stof(argv[++i]);
        else if (arg == "--tepd" && i + 1 < argc) tepd = (std::string(argv[++i]) == "on");
        else if (arg == "--bpp" && i + 1 < argc) bpp = std::stoi(argv[++i]);
        else if (inputPath.empty()) inputPath = arg;
        else if (outputPath.empty()) outputPath = arg;
    }

    if (inputPath.empty() || outputPath.empty()) {
        std::cout << "Usage: image-resizer <in_path> <out_path> [options]" << std::endl;
        std::cout << "Options: --scale 2.0 --algo fsr|lanczos3|bicubic|nearest --bpp 32|24" << std::endl;
        std::cout << "PostFx: --rcas on|off --sharpness 0.2 --rcas-denoise on|off --lfga 0.0 --tepd on|off" << std::endl;
        return 1;
    }

    bool useRcas = (rcasInput == "on" || (rcasInput == "auto" && algo == "fsr"));

    if (fs::is_directory(inputPath)) {
        if (!fs::exists(outputPath)) fs::create_directories(outputPath);
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            if (entry.is_regular_file() && entry.path().extension().string() == ".png") {
                std::string outFilePath = (fs::path(outputPath) / (entry.path().stem().string() + suffix + ".png")).string();
                processImage(entry.path().string(), outFilePath, scale, algo, useRcas, sharpness, rcasDenoise, lfga, tepd, bpp);
            }
        }
    } else {
        processImage(inputPath, outputPath, scale, algo, useRcas, sharpness, rcasDenoise, lfga, tepd, bpp);
    }
    return 0;
}
