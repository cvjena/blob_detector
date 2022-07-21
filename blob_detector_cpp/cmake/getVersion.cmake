set(_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/blob_detector/version.h")

file(STRINGS "${_VERSION_FILE}" _VERSION_PARTS REGEX "#define BLOBDET_VERSION_[A-Z]+[ ]+" )

string(REGEX REPLACE ".+BLOBDET_VERSION_MAJOR[ ]+([0-9]+).*" "\\1" _VERSION_MAJ "${_VERSION_PARTS}")
string(REGEX REPLACE ".+BLOBDET_VERSION_MINOR[ ]+([0-9]+).*" "\\1" _VERSION_MIN "${_VERSION_PARTS}")
string(REGEX REPLACE ".+BLOBDET_VERSION_REVISION[ ]+([0-9]+).*" "\\1" _VERSION_REV "${_VERSION_PARTS}")

set(BlobDetector_VERSION "${_VERSION_MAJ}.${_VERSION_MIN}.${_VERSION_REV}")

set_property(TARGET blob_detector PROPERTY VERSION ${BlobDetector_VERSION})
set_property(TARGET blob_detector PROPERTY SOVERSION 0)
set_property(TARGET blob_detector PROPERTY
  INTERFACE_blob_detector_MAJOR_VERSION 0)
set_property(TARGET blob_detector APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING blob_detector_MAJOR_VERSION
)

message( STATUS "Blob Detector version: ${BlobDetector_VERSION}" )


# create a dependency on the version file
# we never use the output of the following command but cmake will rerun automatically if the version file changes
configure_file("${_VERSION_FILE}" "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/.junk/version.junk" COPYONLY)
