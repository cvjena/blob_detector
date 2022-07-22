cmake_minimum_required(VERSION 3.22)

project( BlobDetectorExample CXX)

include( FetchContent )
FetchContent_Declare( json
    URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz )
FetchContent_MakeAvailable( json )


find_package( OpenCV REQUIRED opencv_imgcodecs )
find_package( BlobDetector REQUIRED )

include_directories(
    ${BLOBDET_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable( detect main.cpp )
target_link_libraries( detect
                       PUBLIC ${BLOBDET_LIBRARIES}
                       PUBLIC opencv_imgcodecs
                       PRIVATE nlohmann_json::nlohmann_json
                      )
