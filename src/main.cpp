#include <iostream>
#include <string>
#include <vector>
#include <filesystem> // C++17 Magic!
#include <algorithm>
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

extern void scaleFSR_EASU(const unsigned char* input, int inW, int inH, unsigned char* output, int outW, int outH);
extern void applyFSR_RCAS(const unsigned char* input, int w, int h, unsigned char* output, float sharpness);

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

// Reusable function to process a single image
bool processImage(const std::string& inFile, const std::string& outFile, float scale, const std::string& algo, bool useRcas, float sharpness) {
    int width, height, channels;
    unsigned char* imgData = stbi_load(inFile.c_str(), &width, &height, &channels, 4);
    if (!imgData) {
        std::cout << "Failed to load: " << inFile << std::endl;
        return false;
    }

    int newW = (int)(width * scale);
    int newH = (int)(height * scale);
    unsigned char* finalData = new unsigned char[newW * newH * 4];

    if (algo == "nearest") {
        scaleNearestNeighbor(imgData, width, height, finalData, newW, newH);
    } 
    else if (algo == "fsr") {
        if (useRcas) {
            unsigned char* easuData = new unsigned char[newW * newH * 4];
            scaleFSR_EASU(imgData, width, height, easuData, newW, newH);
            applyFSR_RCAS(easuData, newW, newH, finalData, sharpness);
            delete[] easuData;
        } else {
            scaleFSR_EASU(imgData, width, height, finalData, newW, newH);
        }
    }

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
    bool useRcas = true;

    // Parse command line flags
    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--scale" && i + 1 < argc) scale = std::stof(argv[++i]);
        else if (arg == "--algo" && i + 1 < argc) algo = argv[++i];
        else if (arg == "--sharpness" && i + 1 < argc) sharpness = std::stof(argv[++i]);
        else if (arg == "--suffix" && i + 1 < argc) suffix = argv[++i];
        else if (arg == "--rcas" && i + 1 < argc) {
            std::string rcasFlag = argv[++i];
            useRcas = (rcasFlag == "on");
        }
        else if (inputPath.empty()) inputPath = arg;
        else if (outputPath.empty()) outputPath = arg;
    }

    if (inputPath.empty() || outputPath.empty()) {
        std::cout << "Usage: image-resizer.exe <input_file_or_folder> <output_file_or_folder> [options]" << std::endl;
        std::cout << "Options: --scale 2.0  --algo fsr|nearest  --rcas on|off  --sharpness 0.2  --suffix _hd" << std::endl;
        return 1;
    }

    // Check if input is a directory (Batch Mode)
    if (fs::is_directory(inputPath)) {
        
        // Safeguard: Output must be a directory
        if (fs::exists(outputPath) && !fs::is_directory(outputPath)) {
            std::cout << "Error: Since input is a folder, output must also be a folder!" << std::endl;
            return 1;
        }

        // Safeguard: Input and Output cannot be the exact same folder
        if (fs::absolute(inputPath) == fs::absolute(outputPath)) {
            std::cout << "Error: Input and Output folders cannot be the exact same directory!" << std::endl;
            return 1;
        }

        // Create output folder if it doesn't exist
        if (!fs::exists(outputPath)) {
            fs::create_directories(outputPath);
        }

        std::cout << "Batch Processing Folder: " << inputPath << std::endl;
        
        // Iterate through files in the folder
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                
                // Convert extension to lowercase for safe checking
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (ext == ".png") {
                    std::string outName = entry.path().stem().string() + suffix + ".png";
                    std::string outFilePath = (fs::path(outputPath) / outName).string();
                    
                    std::cout << " -> " << entry.path().filename().string() << std::endl;
                    processImage(entry.path().string(), outFilePath, scale, algo, useRcas, sharpness);
                }
            }
        }
        std::cout << "Batch Processing Complete!" << std::endl;
    } 
    else {
        // Single File Mode
        std::cout << "Processing Single File..." << std::endl;
        if (processImage(inputPath, outputPath, scale, algo, useRcas, sharpness)) {
            std::cout << "Success!" << std::endl;
        }
    }

    return 0;
}
