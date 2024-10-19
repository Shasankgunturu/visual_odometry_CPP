cmake_minimum_required(VERSION 3.1)
project( OdometryScratch )
set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(/home/shasankgunturu/personal/ComputerVisionBasics/src/include)

add_executable( odometry src/odometry_.cpp )
target_link_libraries( odometry ${OpenCV_LIBS})