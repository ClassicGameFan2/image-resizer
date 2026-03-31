#include "dither.h"
#include "lodepng.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// 1. EXTRACT EXACT ORIGINAL PALETTE
bool loadOriginalPalette(const std::string& filename, std::vector<ColorRGBA>& outPalette, bool& hasTransparency) {
    std::vector<unsigned char> buffer;
    if (lodepng::load_file(buffer, filename)) return false;
    
    lodepng::State state;
    state.decoder.color_convert = 0; 
    
    std::vector<unsigned char> dummyPixels;
    unsigned w, h;
    if (lodepng::decode(dummyPixels, w, h, state, buffer)) return false;
    if (state.info_png.color.colortype != LCT_PALETTE) return false;

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
    
    while (outPalette.size() < 256) outPalette.push_back({0, 0, 0, 255});
    return true;
}

// NEW: 1.5 GENERATE OPTIMAL PALETTE (MEDIAN CUT ALGORITHM)
void generatePalette(const unsigned char* image, int width, int height, std::vector<ColorRGBA>& outPalette, bool& hasTransparency) {
    outPalette.clear();
    hasTransparency = false;
    std::vector<ColorRGBA> pixels;
    pixels.reserve(width * height);

    // Filter out transparency
    for (int i = 0; i < width * height; ++i) {
        ColorRGBA c = { image[i*4], image[i*4+1], image[i*4+2], image[i*4+3] };
        if (c.a < 128) hasTransparency = true;
        else pixels.push_back(c);
    }

    if (hasTransparency) outPalette.push_back({0, 0, 0, 0}); // Reserve Index 0

    if (pixels.empty()) { // Edge case: completely invisible image
        while(outPalette.size() < 256) outPalette.push_back({0,0,0,255});
        return;
    }

    struct Bucket {
        std::vector<ColorRGBA> colors;
        int rMin=255, rMax=0, gMin=255, gMax=0, bMin=255, bMax=0;
        void updateBounds() {
            rMin=255; rMax=0; gMin=255; gMax=0; bMin=255; bMax=0;
            for(const auto& c : colors) {
                if(c.r < rMin) rMin = c.r; if(c.r > rMax) rMax = c.r;
                if(c.g < gMin) gMin = c.g; if(c.g > gMax) gMax = c.g;
                if(c.b < bMin) bMin = c.b; if(c.b > bMax) bMax = c.b;
            }
        }
    };

    std::vector<Bucket> buckets;
    Bucket initial;
    initial.colors = pixels;
    initial.updateBounds();
    buckets.push_back(initial);

    size_t targetColors = hasTransparency ? 255 : 256;

    // Split buckets until we hit our target color count
    while (buckets.size() < targetColors) {
        int maxRange = -1;
        int maxIdx = -1;
        int splitAxis = 0; 

        for (size_t i = 0; i < buckets.size(); ++i) {
            if (buckets[i].colors.size() < 2) continue;
            int dr = buckets[i].rMax - buckets[i].rMin;
            int dg = buckets[i].gMax - buckets[i].gMin;
            int db = buckets[i].bMax - buckets[i].bMin;
            int range = std::max({dr, dg, db});
            if (range > maxRange) {
                maxRange = range; maxIdx = (int)i;
                if (range == dr) splitAxis = 0;
                else if (range == dg) splitAxis = 1;
                else splitAxis = 2;
            }
        }

        if (maxIdx == -1) break; // Cannot split anymore

        Bucket& b = buckets[maxIdx];
        if (splitAxis == 0) std::sort(b.colors.begin(), b.colors.end(), [](const ColorRGBA& c1, const ColorRGBA& c2){ return c1.r < c2.r; });
        else if (splitAxis == 1) std::sort(b.colors.begin(), b.colors.end(), [](const ColorRGBA& c1, const ColorRGBA& c2){ return c1.g < c2.g; });
        else std::sort(b.colors.begin(), b.colors.end(), [](const ColorRGBA& c1, const ColorRGBA& c2){ return c1.b < c2.b; });

        size_t median = b.colors.size() / 2;
        Bucket b1, b2;
        b1.colors.assign(b.colors.begin(), b.colors.begin() + median);
        b2.colors.assign(b.colors.begin() + median, b.colors.end());

        b1.updateBounds(); b2.updateBounds();
        buckets[maxIdx] = b1;
        buckets.push_back(b2);
    }

    // Average the colors in each bucket to generate the final palette
    for (const auto& b : buckets) {
        if (b.colors.empty()) continue;
        long long rSum = 0, gSum = 0, bSum = 0;
        for (const auto& c : b.colors) { rSum += c.r; gSum += c.g; bSum += c.b; }
        outPalette.push_back({
            (unsigned char)(rSum / b.colors.size()),
            (unsigned char)(gSum / b.colors.size()),
            (unsigned char)(bSum / b.colors.size()), 255
        });
    }

    while (outPalette.size() < 256) outPalette.push_back({0, 0, 0, 255});
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
    if (error) return false;
    error = lodepng::save_file(buffer, filename);
    return error == 0;
}
