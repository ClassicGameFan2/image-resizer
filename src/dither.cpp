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
    // CRITICAL FIX: Tell the decoder NOT to convert the color format to 32-bit so we can read the raw palette!
    state.decoder.color_convert = 0; 
    
    std::vector<unsigned char> dummyPixels;
    unsigned w, h;
    unsigned error = lodepng::decode(dummyPixels, w, h, state, buffer);
    if (error) {
        std::cout << "LodePNG decode error: " << lodepng_error_text(error) << std::endl;
        return false;
    }

    if (state.info_png.color.colortype != LCT_PALETTE) {
        return false;
    }

    outPalette.clear();
    hasTransparency = false;
    size_t palSize = state.info_png.color.palettesize;
    
    // Rip the exact original palette
    for (size_t i = 0; i < palSize; ++i) {
        ColorRGBA c;
        c.r = state.info_png.color.palette[i * 4 + 0];
        c.g = state.info_png.color.palette[i * 4 + 1];
        c.b = state.info_png.color.palette[i * 4 + 2];
        c.a = state.info_png.color.palette[i * 4 + 3];
        outPalette.push_back(c);
        if (c.a < 128) hasTransparency = true;
    }
    
    // Pad to exactly 256 colors to prevent retro engine crashes
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
            
            if (hasTransparency && input[pIdx+3] < 128 && transpIdx != -1) {
                output[idx] = (unsigned char)transpIdx; 
                continue; 
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
    
    // CRITICAL FIX: Force LodePNG to save exactly as 8-bit, do not auto-compress to 1-bit!
    state.encoder.auto_convert = 0; 
    
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
    if (error) {
        std::cout << "LodePNG encode error: " << lodepng_error_text(error) << std::endl;
        return false;
    }
    
    error = lodepng::save_file(buffer, filename);
    return error == 0;
}
