cmake_minimum_required(VERSION 3.22)

project( BlobDetectorExample CXX)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( detect main.cpp )

target_link_libraries( detect ${OpenCV_LIBS} )
