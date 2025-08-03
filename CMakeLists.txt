cmake_minimum_required(VERSION 3.10)
project(3DGS)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(3DGS src/depth2pcd.cpp)
target_link_libraries(3DGS ${OpenCV_LIBS})
