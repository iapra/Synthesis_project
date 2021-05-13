cmake_minimum_required(VERSION 3.17)
project(Roofs)

set(CMAKE_CXX_STANDARD 14)

include_directories(external/KDTree)
include_directories(external/linalg)
include_directories(external/nlohmann-json)
include_directories(src)

add_executable(Roofs
        external/KDTree/kdtree.h
        external/KDTree/sample.cpp
        external/linalg/linalg.h
        external/linalg/README.md
        external/linalg/UNLICENSE
        external/nlohmann-json/json.hpp
        src/main.cpp
        src/roofs.cpp
        src/roofs.h)
