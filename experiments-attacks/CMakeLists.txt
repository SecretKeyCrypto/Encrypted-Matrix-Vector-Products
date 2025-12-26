# Copyright (c) 2025. Amazon.com, Inc. or its affiliates. All Rights Reserved.
cmake_minimum_required(VERSION 3.22)
project(EMVPExpr)

set(CMAKE_CXX_STANDARD 20)

# Add cmake modules directory to module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find GMP for NTL
find_package(GMP REQUIRED)
get_filename_component(GMP_LIBRARY_DIR ${GMP_LIBRARY} DIRECTORY)

# Find NTL package
find_package(NTL REQUIRED)

# Create library from utils and prvector modules
add_library(emvptools STATIC
    src/utils.cpp
    src/prvector.cpp
)

target_include_directories(emvptools PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${NTL_INCLUDE_DIR}
    ${GMP_INCLUDE_DIR}
)

target_link_libraries(emvptools PUBLIC
    ${NTL_LIBRARIES}
    GMP::GMP
)

# Create executable
add_executable(algebraic-attacks src/main-attacks.cpp)
target_link_libraries(algebraic-attacks PRIVATE emvptools)
