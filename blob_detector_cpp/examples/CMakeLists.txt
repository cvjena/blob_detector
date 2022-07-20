cmake_minimum_required(VERSION 3.22)

project( BlobDetectorExample CXX)

find_package( OpenCV REQUIRED )

include_directories(
    ../include
    ${OpenCV_INCLUDE_DIRS}
)

add_library( blob_detector SHARED
             ../src/bbox.cpp
             ../src/bbox_ops.cpp
             ../src/image_ops.cpp
             ../src/utils.cpp
            )

target_link_libraries( blob_detector
                       ${OpenCV_LIBS} )


add_executable( detect main.cpp )
target_link_libraries( detect
                       PUBLIC blob_detector
                       PUBLIC ${OpenCV_LIBS} )
