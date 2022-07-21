cmake_minimum_required(VERSION 3.22)

project( BlobDetectorExample CXX)

find_package( OpenCV REQUIRED )
find_package( BlobDetector REQUIRED )

include_directories(
    ../include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable( detect main.cpp )
target_link_libraries( detect
                       ${BLOBDET_LIBRARIES}
                       ${OpenCV_LIBS}
                      )
