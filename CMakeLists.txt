cmake_minimum_required(VERSION 3.10)
project(ImageResizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Neat Trick: Auto-download the stb_image library files so we don't have to copy/paste them!
set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
file(DOWNLOAD "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" "${STB_DIR}/stb_image.h")
file(DOWNLOAD "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" "${STB_DIR}/stb_image_write.h")
include_directories(${STB_DIR})

# Notice we added src/stb.cpp to the list of files to compile!
add_executable(image-resizer src/main.cpp src/stb.cpp)

if(MSVC)
    set_property(TARGET image-resizer PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
