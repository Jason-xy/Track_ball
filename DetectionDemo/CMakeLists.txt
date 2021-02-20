cmake_minimum_required(VERSION 2.8)

project(detection)

find_package(jetson-utils)
find_package(jetson-inference)
find_package(CUDA)
find_package(Qt4)

include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

cuda_add_executable(detection detection.cpp)
target_link_libraries(detection jetson-inference)