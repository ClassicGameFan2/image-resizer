cmake_minimum_required(VERSION 3.10)
project(ImageResizer)

# Tell it to use standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Create our executable
add_executable(image-resizer src/main.cpp)

# Statically link the Windows runtime so it works on Windows 7 without missing DLLs
if(MSVC)
    set_property(TARGET image-resizer PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
