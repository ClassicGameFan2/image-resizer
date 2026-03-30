#pragma once
#include <vector>

struct ColorRGBA {
    unsigned char r, g, b, a;
    bool operator==(const ColorRGBA& o) const {
        return r == o.r && g == o.g && b == o.b && a == o.a;
    }
};

bool extractPalette(const unsigned char* input, int width, int height, std::vector<ColorRGBA>& outPalette, bool& hasTransparency);

void quantizeAndDither(const unsigned char* input, int width, int height, unsigned char* output, 
                       const std::vector<ColorRGBA>& palette, bool hasTransparency, bool useFloydSteinberg);

bool saveIndexedPNG(const char* filename, const unsigned char* indexedData, int width, int height, const std::vector<ColorRGBA>& palette);
