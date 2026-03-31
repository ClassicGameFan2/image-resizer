#include "dither.h"
#include "lodepng.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// 1. EXTRACT EXACT ORIGINAL PALETTE USING LODEPNG
bool loadOriginalPalette(const std::string& filename, std::vector<ColorRGBA>& outPalette, bool& hasTransparency) {
    std::vector<unsigned char> buffer;
    lodepng::load_file(buffer, filename);
    
    lodepng::State state;
    unsigned w, h;
    unsigned error = lodepng_inspect(&w, &h, &state, buffer.data(), buffer.size());
    if (error) return false;

    // Ensure the input image actually has a palette!
    if (state.info_png.color.colortype != LCT_PALETTE) {
        return false;
    }

    outPalette.clear();
    hasTransparency = false;
    size_t palSize = state.info_png.color.palettesize;
    
    for (size_t i = 0; i < palSize; ++i) {
        ColorRGBA c;
        c.r = state.info_png.color.palette[i * 4 + 0];
        c.g = state.info_png.color.palette[i * 4 + 1];
        c.b = state.info_png.color.palette[i * 4 + 2];
        c.a = state.info_png.color.palette[i * 4 + 3];
        outPalette.push_back(c);
        if (c.a < 128) hasTransparency = true;
    }
    
    // Pad the palette to exactly 256 colors. Retro engines crash if it's less than 256!
    while (outPalette.size() < 256) {
        outPalette.push_back({0, 0, 0, 255});
    }

    return true;
}

// 2. FIND NEAREST SOLID COLOR
int findNearestSolid(int r, int g, int b, const std::vector<ColorRGBA>& palette) {
    int bestIdx = 0;
    int bestDist = 255 * 255 * 3 + 1;
    
    for (size_t i = 0; i < palette.size(); ++i) {
        // Skip transparent indices so we don't accidentally snap solid colors to invisibility
        if (palette[i].a < 128) continue; 
        
        int dr = r - palette[i].r;
        int dg = g - palette[i].g;
        int db = b - palette[i].b;
        int dist = dr*dr + dg*dg + db*db;
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
    
    std::vector<float> errBuf(width * height * 3, 0.0f);
    
    // Find exactly where the transparent color lives in the original palette
    int transpIdx = -1;
    if (hasTransparency) {
        for (size_t i = 0; i < palette.size(); ++i) {
            if (palette[i].a < 128) { transpIdx = (int)i; break; }
        }
    }
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x);
            int pIdx = idx * 4;
            
            // If the pixel is transparent, snap it to the exact original transparency index!
            if (hasTransparency && input[pIdx+3] < 128 && transpIdx != -1) {
                output[idx] = (unsigned char)transpIdx; 
                continue; // Do not diffuse error for transparent pixels
            }
            
            float r = input[pIdx+0] + errBuf[idx*3+0];
            float g = input[pIdx+1] + errBuf[idx*3+1];
            float b = input[pIdx+2] + errBuf[idx*3+2];
            
            r = std::max(0.0f, std::min(255.0f, r));
            g = std::max(0.0f, std::min(255.0f, g));
            b = std::max(0.0f, std::min(255.0f, b));
            
            int palIdx = findNearestSolid((int)r, (int)g, (int)b, palette);
            output[idx] = (unsigned char)palIdx;
            
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
    
    state.info_png.color.colortype = LCT_PALETTE;
    state.info_png.color.bitdepth = 8;
    state.info_raw.colortype = LCT_PALETTE;
    state.info_raw.bitdepth = 8;
    
    for (const auto& c : palette) {
        lodepng_palette_add(&state.info_png.color, c.r, c.g, c.b, c.a);
        lodepng_palette_add(&state.info_raw, c.r, c.g, c.b, c.a);
    }
    
    std::vector<unsigned char> buffer;
    unsigned error = lodepng::encode(buffer, indexedData, width, height, state);
    if (error) return false;
    
    error = lodepng::save_file(buffer, filename);
    return error == 0;
}
