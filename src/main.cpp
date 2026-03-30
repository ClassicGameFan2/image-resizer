#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cstring> // Needed for memcpy
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

// FSR
extern void scaleFSR_EASU(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH);
extern void applyFSR_RCAS(const unsigned char* input, int w, int h, unsigned char* output, float sharpness);

// Standard Scalers
extern void scaleNearestNeighbor(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH);
extern void scaleBilinear(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH);
extern void scaleBicubic(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH);
extern void scaleLanczos3(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH);

bool processImage(const std::string& inFile, const std::string& outFile, float scale, const std::string& algo, bool useRcas, float sharpness) {
    int width, height, channels;
    unsigned char* imgData = stbi_load(inFile.c_str(), &width, &height, &channels, 4);
    if (!imgData) {
        std::cout << "Failed to load: " << inFile << std::endl;
        return false;
    }

    // Step 1: Base Resolution Logic
    int newW = width;
    int newH = height;
    
    if (algo != "off") {
        newW = std::max(1, (int)(width * scale));
        newH = std::max(1, (int)(height * scale));
    }

    unsigned char* step1Data = new unsigned char[newW * newH * 4];

    // Step 2: The Scaling Pass
    if (algo == "off") {
        std::memcpy(step1Data, imgData, newW * newH * 4); // Just copy the original pixels
    }
    else if (algo == "nearest") {
        scaleNearestNeighbor(imgData, width, height, step1Data, newW, newH);
    } 
    else if (algo == "bilinear") {
        scaleBilinear(imgData, width, height, step1Data, newW, newH);
    }
    else if (algo == "bicubic") {
        scaleBicubic(imgData, width, height, step1Data, newW, newH);
    }
    else if (algo == "lanczos3") {
        scaleLanczos3(imgData, width, height, step1Data, newW, newH);
    }
    else if (algo == "fsr") {
        if (scale < 1.0f) {
            std::cout << "Error: FSR 1.0 mathematically cannot downscale. Please use --algo bicubic or lanczos3." << std::endl;
            stbi_image_free(imgData);
            delete[] step1Data;
            return false;
        }
        scaleFSR_EASU(imgData, width, height, step1Data, newW, newH);
    } 
    else {
        std::cout << "Unknown algorithm: " << algo << std::endl;
        stbi_image_free(imgData);
        delete[] step1Data;
        return false;
    }

    // Step 3: The Optional RCAS Sharpening Pass
    unsigned char* finalData = step1Data; // Assume no RCAS initially

    if (useRcas) {
        finalData = new unsigned char[newW * newH * 4];
        applyFSR_RCAS(step1Data, newW, newH, finalData, sharpness);
        delete[] step1Data; // Free the unsharpened memory
    }

    // Step 4: Save
    stbi_write_png(outFile.c_str(), newW, newH, 4, finalData, newW * 4);

    stbi_image_free(imgData);
    delete[] finalData;
    return true;
}

int main(int argc, char** argv) {
    std::cout << "--- Image Resizer CPU ---" << std::endl;

    std::string inputPath = "";
    std::string outputPath = "";
    std::string suffix = "";
    float scale = 2.0f;
    float sharpness = 0.2f;
    std::string algo = "fsr";
    std::string rcasInput = "auto"; // Auto means ON for FSR, OFF for everything else

    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--scale" && i + 1 < argc) scale = std::stof(argv[++i]);
        else if (arg == "--algo" && i + 1 < argc) algo = argv[++i];
        else if (arg == "--sharpness" && i + 1 < argc) sharpness = std::stof(argv[++i]);
        else if (arg == "--suffix" && i + 1 < argc) suffix = argv[++i];
        else if (arg == "--rcas" && i + 1 < argc) rcasInput = argv[++i];
        else if (inputPath.empty()) inputPath = arg;
        else if (outputPath.empty()) outputPath = arg;
    }

    if (inputPath.empty() || outputPath.empty()) {
        std::cout << "Usage: image-resizer.exe <in_path> <out_path> [options]" << std::endl;
        std::cout << "Options: --scale 2.0 --algo fsr|nearest|bilinear|bicubic|lanczos3|off --rcas on|off --sharpness 0.2 --suffix _hd" << std::endl;
        return 1;
    }

    // Determine RCAS behavior
    bool useRcas = false;
    if (rcasInput == "on") useRcas = true;
    else if (rcasInput == "off") useRcas = false;
    else if (algo == "fsr") useRcas = true; // Auto behavior: FSR defaults to RCAS ON

    if (fs::is_directory(inputPath)) {
        if (fs::exists(outputPath) && !fs::is_directory(outputPath)) {
            std::cout << "Error: Output must be a folder!" << std::endl;
            return 1;
        }
        if (fs::absolute(inputPath) == fs::absolute(outputPath)) {
            std::cout << "Error: Input and Output folders cannot be the same!" << std::endl;
            return 1;
        }
        if (!fs::exists(outputPath)) fs::create_directories(outputPath);

        std::cout << "Batch Processing Folder: " << inputPath << std::endl;
        
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (ext == ".png") {
                    std::string outName = entry.path().stem().string() + suffix + ".png";
                    std::string outFilePath = (fs::path(outputPath) / outName).string();
                    std::cout << " -> " << entry.path().filename().string() << std::endl;
                    processImage(entry.path().string(), outFilePath, scale, algo, useRcas, sharpness);
                }
            }
        }
        std::cout << "Batch Complete!" << std::endl;
    } 
    else {
        std::cout << "Processing Single File..." << std::endl;
        if (processImage(inputPath, outputPath, scale, algo, useRcas, sharpness)) {
            std::cout << "Success!" << std::endl;
        }
    }

    return 0;
}
