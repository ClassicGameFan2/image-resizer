#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"

int main(int argc, char** argv) {
    std::cout << "--- Image Resizer ---" << std::endl;

    // We need an input and output file name from the command line
    if (argc < 3) {
        std::cout << "Usage: image-resizer.exe <input.png> <output.png>" << std::endl;
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];

    std::cout << "Loading: " << inputFile << "..." << std::endl;

    int width, height, channels;
    // We force loading as 4 channels (RGBA) because FSR math requires it later
    unsigned char* imgData = stbi_load(inputFile, &width, &height, &channels, 4);

    if (!imgData) {
        std::cout << "Failed to load image! Reason: " << stbi_failure_reason() << std::endl;
        return 1;
    }

    std::cout << "Success! Image is " << width << "x" << height << " with " << channels << " original channels." << std::endl;
    
    // For now, let's just save an exact duplicate copy to prove we can write PNGs
    std::cout << "Saving exact copy to: " << outputFile << "..." << std::endl;
    int stride = width * 4; // 4 bytes per pixel (Red, Green, Blue, Alpha)
    int success = stbi_write_png(outputFile, width, height, 4, imgData, stride);

    // Free the memory
    stbi_image_free(imgData);

    if (success) {
        std::cout << "Image saved successfully!" << std::endl;
    } else {
        std::cout << "Failed to save image." << std::endl;
        return 1;
    }

    return 0;
}
