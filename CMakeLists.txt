cmake_minimum_required(VERSION 3.5)
project( OdometryScratch )
set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(include/)

add_executable( odometry src/odometry_.cpp )
target_link_libraries( odometry ${OpenCV_LIBS})
