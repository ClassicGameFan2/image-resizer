#include <iostream>
#include <string>
#include "stb_image.h"
#include "stb_image_write.h"

// Our simple Nearest Neighbor scaling function
void scaleNearestNeighbor(const unsigned char* input, int inW, int inH, 
                          unsigned char* output, int outW, int outH) {
    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            // Find the closest pixel in the original image
            int srcX = (x * inW) / outW;
            int srcY = (y * inH) / outH;

            // Calculate memory addresses
            int srcIndex = (srcY * inW + srcX) * 4;
            int dstIndex = (y * outW + x) * 4;

            // Copy the Red, Green, Blue, and Alpha bytes
            output[dstIndex + 0] = input[srcIndex + 0];
            output[dstIndex + 1] = input[srcIndex + 1];
            output[dstIndex + 2] = input[srcIndex + 2];
            output[dstIndex + 3] = input[srcIndex + 3];
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "--- Image Resizer ---" << std::endl;

    // We now require 3 arguments: input, output, and scale factor
    if (argc < 4) {
        std::cout << "Usage: image-resizer.exe <input.png> <output.png> <scale>" << std::endl;
        std::cout << "Example: image-resizer.exe hero.png hero_hd.png 2.0" << std::endl;
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];
    float scale = std::stof(argv[3]); // Convert the text "2.0" into a math number

    std::cout << "Loading: " << inputFile << "..." << std::endl;

    int width, height, channels;
    unsigned char* imgData = stbi_load(inputFile, &width, &height, &channels, 4);

    if (!imgData) {
        std::cout << "Failed to load image!" << std::endl;
        return 1;
    }

    // Calculate the new resolution
    int newWidth = (int)(width * scale);
    int newHeight = (int)(height * scale);

    std::cout << "Original Size: " << width << "x" << height << std::endl;
    std::cout << "Scaling to: " << newWidth << "x" << newHeight << " (Scale: " << scale << "x)" << std::endl;

    // Create a new blank canvas in memory for the big image
    unsigned char* outputData = new unsigned char[newWidth * newHeight * 4];

    // Run our scaling algorithm!
    std::cout << "Processing (Nearest Neighbor)..." << std::endl;
    scaleNearestNeighbor(imgData, width, height, outputData, newWidth, newHeight);

    // Save the new big image
    std::cout << "Saving: " << outputFile << "..." << std::endl;
    int stride = newWidth * 4;
    int success = stbi_write_png(outputFile, newWidth, newHeight, 4, outputData, stride);

    // Clean up memory so we don't crash the computer!
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
