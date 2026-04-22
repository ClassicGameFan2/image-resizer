cmake_minimum_required(VERSION 3.10)
project(ImageResizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Auto-download stb_image
file(DOWNLOAD "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" "${STB_DIR}/stb_image.h")
file(DOWNLOAD "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" "${STB_DIR}/stb_image_write.h")

# Auto-download LodePNG for 8-bit palette support
file(DOWNLOAD "https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.h" "${STB_DIR}/lodepng.h")
file(DOWNLOAD "https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.cpp" "${STB_DIR}/lodepng.cpp")

include_directories(${STB_DIR})

add_executable(image-resizer 
    src/main.cpp 
    src/stb.cpp 
    src/fsr_easu.cpp 
    src/fsr_rcas.cpp 
    src/scalers.cpp
    src/dither.cpp
    # FSR 2.3.4 CPU Port
    src/fsr2_jitter_resample.cpp
    src/fsr2_pass_luminance.cpp
    src/fsr2_pass_reconstruct_depth.cpp
    src/fsr2_pass_depth_clip.cpp
    src/fsr2_pass_lock.cpp
    src/fsr2_pass_accumulate.cpp
    src/fsr2_pass_rcas.cpp
    src/fsr2_pass_reactive.cpp
    src/fsr2_context.cpp
    src/fsr2_main.cpp
    third_party/lodepng.cpp
)

if(MSVC)
    set_property(TARGET image-resizer PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
