cmake_minimum_required(VERSION 3.10)

project("AUTOMOTIVE_AI")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(ONNXRUNTIME_ROOT "/opt/onnxruntime")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

include_directories(include)

add_executable(app src/main.cpp src/object_detector.cpp src/tracker.cpp src/lane_detector.cpp)
target_link_libraries(app ${OpenCV_LIBS} onnxruntime)