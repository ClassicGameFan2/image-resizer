#pragma once
#include <vector>
#include <string>

struct ColorRGBA {
    unsigned char r, g, b, a;
    bool operator==(const ColorRGBA& o) const {
        return r == o.r && g == o.g && b == o.b && a == o.a;
    }
};

bool loadOriginalPalette(const std::string& filename, std::vector<ColorRGBA>& outPalette, bool& hasTransparency);

// NEW: Generates an optimal 256-color palette from a 32-bit image!
void generatePalette(const unsigned char* image, int width, int height, std::vector<ColorRGBA>& outPalette, bool& hasTransparency);

void quantizeAndDither(const unsigned char* input, int width, int height, unsigned char* output, 
                       const std::vector<ColorRGBA>& palette, bool hasTransparency, bool useFloydSteinberg);

bool saveIndexedPNG(const char* filename, const unsigned char* indexedData, int width, int height, const std::vector<ColorRGBA>& palette);
