cmake_minimum_required(VERSION 3.22)

project( BlobDetectorExample CXX)

find_package( OpenCV REQUIRED opencv_highgui )
find_package( BlobDetector REQUIRED )

include_directories(
    ${BLOBDET_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable( detect main.cpp )
target_link_libraries( detect
                       ${BLOBDET_LIBRARIES}
                       opencv_highgui
                      )
