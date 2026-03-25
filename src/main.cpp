#include <iostream>
#include <string>
#include "stb_image.h"
#include "stb_image_write.h"

// Bring in both FSR functions!
extern void scaleFSR_EASU(const unsigned char* input, int inW, int inH, 
                          unsigned char* output, int outW, int outH);
                          
extern void applyFSR_RCAS(const unsigned char* input, int w, int h, 
                          unsigned char* output, float sharpness);

int main(int argc, char** argv) {
    std::cout << "--- Image Resizer (AMD FSR 1.0 CPU Port) ---" << std::endl;

    if (argc < 4) {
        std::cout << "Usage: image-resizer.exe <input.png> <output.png> <scale> [sharpness]" << std::endl;
        std::cout << "Sharpness: 0.0 (Max Sharp) to 2.0 (Soft). Default is 0.2" << std::endl;
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];
    float scale = std::stof(argv[3]);
    
    // Set default sharpness to 0.2, but allow the user to override it!
    float sharpness = 0.2f;
    if (argc > 4) {
        sharpness = std::stof(argv[4]);
    }

    std::cout << "Loading: " << inputFile << "..." << std::endl;

    int width, height, channels;
    unsigned char* imgData = stbi_load(inputFile, &width, &height, &channels, 4);

    if (!imgData) {
        std::cout << "Failed to load image!" << std::endl;
        return 1;
    }

    int newW = (int)(width * scale);
    int newH = (int)(height * scale);

    std::cout << "Scaling to: " << newW << "x" << newH << " (Scale: " << scale << "x)" << std::endl;
    std::cout << "Sharpness level: " << sharpness << std::endl;

    // Canvas 1: The smoothed EASU image
    unsigned char* easuData = new unsigned char[newW * newH * 4];
    
    // Canvas 2: The final sharpened RCAS image
    unsigned char* finalData = new unsigned char[newW * newH * 4];

    // PASS 1: Scale and Smooth
    std::cout << "Pass 1: FSR EASU (Scaling)..." << std::endl;
    scaleFSR_EASU(imgData, width, height, easuData, newW, newH);

    // PASS 2: Sharpen and Restore Texture
    std::cout << "Pass 2: FSR RCAS (Sharpening)..." << std::endl;
    applyFSR_RCAS(easuData, newW, newH, finalData, sharpness);

    std::cout << "Saving: " << outputFile << "..." << std::endl;
    int stride = newW * 4;
    int success = stbi_write_png(outputFile, newW, newH, 4, finalData, stride);

    // Free all three memory blocks
    stbi_image_free(imgData);
    delete[] easuData;
    delete[] finalData;

    if (success) {
        std::cout << "Image scaled and saved successfully!" << std::endl;
    } else {
        std::cout << "Failed to save image." << std::endl;
        return 1;
    }

    return 0;
}
