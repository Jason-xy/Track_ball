cmake_minimum_required(VERSION 2.8)

project(trackball)

find_package(jetson-utils)
find_package(jetson-inference)
find_package(CUDA)
find_package(OpenCV REQUIRED)
find_package(Qt4)

include_directories(${OpenCV_INCLUDE_DIRS})
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

cuda_add_executable(trackball trackball.cpp)

target_link_libraries(trackball ${OpenCV_LIBS})
target_link_libraries(trackball jetson-inference)