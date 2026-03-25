#include <iostream>
#include <string>
#include "stb_image.h"
#include "stb_image_write.h"

// We tell main.cpp that our FSR function exists in the other file!
extern void scaleFSR_EASU(const unsigned char* input, int inW, int inH, 
                          unsigned char* output, int outW, int outH);

int main(int argc, char** argv) {
    std::cout << "--- Image Resizer (AMD FSR 1.0 CPU Port) ---" << std::endl;

    if (argc < 4) {
        std::cout << "Usage: image-resizer.exe <input.png> <output.png> <scale>" << std::endl;
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];
    float scale = std::stof(argv[3]);

    std::cout << "Loading: " << inputFile << "..." << std::endl;

    int width, height, channels;
    unsigned char* imgData = stbi_load(inputFile, &width, &height, &channels, 4);

    if (!imgData) {
        std::cout << "Failed to load image!" << std::endl;
        return 1;
    }

    int newWidth = (int)(width * scale);
    int newHeight = (int)(height * scale);

    std::cout << "Original Size: " << width << "x" << height << std::endl;
    std::cout << "Scaling to: " << newWidth << "x" << newHeight << " (Scale: " << scale << "x)" << std::endl;

    unsigned char* outputData = new unsigned char[newWidth * newHeight * 4];

    // BOOM! We run the FSR 1.0 math instead of Nearest Neighbor!
    std::cout << "Processing (FSR 1.0 EASU)..." << std::endl;
    scaleFSR_EASU(imgData, width, height, outputData, newWidth, newHeight);

    std::cout << "Saving: " << outputFile << "..." << std::endl;
    int stride = newWidth * 4;
    int success = stbi_write_png(outputFile, newWidth, newHeight, 4, outputData, stride);

    stbi_image_free(imgData);
    delete[] outputData;

    if (success) {
        std::cout << "Image scaled and saved successfully!" << std::endl;
    } else {
        std::cout << "Failed to save image." << std::endl;
        return 1;
    }

    return 0;
}
