#include "dither.h"
#include "lodepng.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// 1. EXTRACT PALETTE & AUTO-DETECT TRANSPARENCY
bool extractPalette(const unsigned char* input, int width, int height, std::vector<ColorRGBA>& outPalette, bool& hasTransparency) {
    outPalette.clear();
    hasTransparency = false;

    for (int i = 0; i < width * height; ++i) {
        ColorRGBA c = { input[i*4], input[i*4+1], input[i*4+2], input[i*4+3] };
        
        if (c.a < 128) {
            if (!hasTransparency) {
                hasTransparency = true;
                // Force the transparent color to be at Index 0
                outPalette.insert(outPalette.begin(), c); 
            }
        } else {
            bool found = false;
            for (const auto& pc : outPalette) {
                if (pc == c) { found = true; break; }
            }
            if (!found) {
                if (outPalette.size() < 256) {
                    outPalette.push_back(c);
                }
            }
        }
    }
    return true;
}

// 2. FIND NEAREST COLOR IN 3D RGB SPACE
int findNearest(int r, int g, int b, const std::vector<ColorRGBA>& palette, int startIdx) {
    int bestIdx = startIdx;
    int bestDist = 255 * 255 * 3 + 1;
    
    for (size_t i = startIdx; i < palette.size(); ++i) {
        int dr = r - palette[i].r;
        int dg = g - palette[i].g;
        int db = b - palette[i].b;
        int dist = dr*dr + dg*dg + db*db; // Euclidean distance
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = (int)i;
        }
    }
    return bestIdx;
}

// 3. THE FLOYD-STEINBERG DITHERING ENGINE
void quantizeAndDither(const unsigned char* input, int width, int height, unsigned char* output, 
                       const std::vector<ColorRGBA>& palette, bool hasTransparency, bool useFloydSteinberg) {
    
    // We use a float buffer to carry the mathematical "error" to neighboring pixels
    std::vector<float> errBuf(width * height * 3, 0.0f);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x);
            int pIdx = idx * 4;
            
            // If the pixel is transparent, snap to Index 0 and skip dithering!
            if (hasTransparency && input[pIdx+3] < 128) {
                output[idx] = 0; 
                continue;
            }
            
            // Add any error pushed to this pixel from previous pixels
            float r = input[pIdx+0] + errBuf[idx*3+0];
            float g = input[pIdx+1] + errBuf[idx*3+1];
            float b = input[pIdx+2] + errBuf[idx*3+2];
            
            r = std::max(0.0f, std::min(255.0f, r));
            g = std::max(0.0f, std::min(255.0f, g));
            b = std::max(0.0f, std::min(255.0f, b));
            
            // Skip index 0 if it is reserved for transparency
            int startIdx = hasTransparency ? 1 : 0; 
            int palIdx = findNearest((int)r, (int)g, (int)b, palette, startIdx);
            output[idx] = (unsigned char)palIdx;
            
            // Calculate the error and diffuse it
            if (useFloydSteinberg) {
                float er = r - palette[palIdx].r;
                float eg = g - palette[palIdx].g;
                float eb = b - palette[palIdx].b;
                
                auto addErr = [&](int nx, int ny, float factor) {
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int nIdx = (ny * width + nx) * 3;
                        errBuf[nIdx+0] += er * factor;
                        errBuf[nIdx+1] += eg * factor;
                        errBuf[nIdx+2] += eb * factor;
                    }
                };
                
                addErr(x+1, y,   7.0f/16.0f);
                addErr(x-1, y+1, 3.0f/16.0f);
                addErr(x,   y+1, 5.0f/16.0f);
                addErr(x+1, y+1, 1.0f/16.0f);
            }
        }
    }
}

// 4. SAVE USING LODEPNG
bool saveIndexedPNG(const char* filename, const unsigned char* indexedData, int width, int height, const std::vector<ColorRGBA>& palette) {
    lodepng::State state;
    
    // Tell LodePNG we want an 8-bit palette
    state.info_png.color.colortype = LCT_PALETTE;
    state.info_png.color.bitdepth = 8;
    state.info_raw.colortype = LCT_PALETTE;
    state.info_raw.bitdepth = 8;
    
    for (const auto& c : palette) {
        lodepng_palette_add(&state.info_png.color, c.r, c.g, c.b, c.a);
        lodepng_palette_add(&state.info_raw.color, c.r, c.g, c.b, c.a);
    }
    
    std::vector<unsigned char> buffer;
    unsigned error = lodepng::encode(buffer, indexedData, width, height, state);
    if (error) return false;
    
    error = lodepng::save_file(buffer, filename);
    return error == 0;
}
