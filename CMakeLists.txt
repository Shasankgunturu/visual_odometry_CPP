cmake_minimum_required(VERSION 3.5)
project(voScratch)

set(CMAKE_CXX_STANDARD 14)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Include your own headers
include_directories(include/)

add_executable(vo_play src/playVO.cpp)

# Link libraries
target_link_libraries(vo_play ${OpenCV_LIBS} ${PCL_LIBRARIES})
