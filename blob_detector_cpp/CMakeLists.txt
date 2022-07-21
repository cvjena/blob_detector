cmake_minimum_required(VERSION 3.22)

project( BlobDetector CXX)

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/cmake/BlobDetector)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()



##### shared library definition #####
find_package( OpenCV REQUIRED )

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

add_library( blob_detector SHARED
             src/bbox.cpp
             src/bbox_ops.cpp
             src/image_ops.cpp
             src/utils.cpp

             include/blob_detector/bbox.h
             include/blob_detector/core.h
             include/blob_detector/operations.h
             include/blob_detector/utils.h
             include/blob_detector/version.h
            )

target_link_libraries( blob_detector ${OpenCV_LIBS} )

set_target_properties(blob_detector PROPERTIES
  PUBLIC_HEADER "include/blob_detector/bbox.h;include/blob_detector/core.h;include/blob_detector/operations.h;include/blob_detector/utils.h;include/blob_detector/version.h")

install(TARGETS blob_detector
  # IMPORTANT: Add the foo library to the "export-set"
  EXPORT BlobDetectorTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/blob_detector"
    COMPONENT dev)

########################################

include(cmake/getVersion.cmake)

message( STATUS "Installation location: ${CMAKE_INSTALL_PREFIX}" )


##### cmake package definition #####

# Add all targets to the build-tree export set
export(TARGETS blob_detector
  FILE "${PROJECT_BINARY_DIR}/BlobDetectorTargets.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE BlobDetector)

# Create the BlobDetectorConfig.cmake and BlobDetectorConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")

# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(cmake/BlobDetectorConfig.cmake.in
  "${PROJECT_BINARY_DIR}/BlobDetectorConfig.cmake" @ONLY)

# ... for the install tree
set(CONF_INCLUDE_DIRS "\${BLOBDET_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake/BlobDetectorConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/BlobDetectorConfig.cmake" @ONLY)

# ... for both
configure_file(cmake/BlobDetectorConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/BlobDetectorConfigVersion.cmake" @ONLY)

# Install the BlobDetectorConfig.cmake and BlobDetectorConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/BlobDetectorConfig.cmake"
  "${PROJECT_BINARY_DIR}/BlobDetectorConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Install the export set for use with the install-tree
install(EXPORT BlobDetectorTargets DESTINATION
  "${INSTALL_CMAKE_DIR}" COMPONENT dev)

########################################


# quit()








# #generate_export_header( blob_detector )

# include(cmake/BlobDetectorVersion.cmake)

# export(TARGETS blob_detector
#   FILE "${PROJECT_BINARY_DIR}/BlobDetectorTargets.cmake")

# export(PACKAGE BlobDetector)


# install(
# 	TARGETS blob_detector
# 	EXPORT BlobDetectorTargets
# 	LIBRARY DESTINATION lib
# 	ARCHIVE DESTINATION lib
# 	RUNTIME DESTINATION bin
# 	INCLUDES DESTINATION include
# )

# install(
# 	FILES
# 		include/blob_detector/core.h
# 		include/blob_detector/bbox.h
# 		include/blob_detector/operations.h
# 		include/blob_detector/utils.h
# 	TYPE INCLUDE
# )

# # Make relative paths absolute (needed later on)
# foreach(p LIB BIN INCLUDE CMAKE)
#   set(var INSTALL_${p}_DIR)
#   if(NOT IS_ABSOLUTE "${${var}}")
#     set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
#   endif()
# endforeach()


# write_basic_package_version_file(
#   "${CMAKE_CURRENT_BINARY_DIR}/blob_detector/BlobDetectorConfigVersion.cmake"
#   VERSION ${BlobDetector_VERSION}
#   COMPATIBILITY AnyNewerVersion
# )

# # configure_file(cmake/BlobDetectorConfig.cmake.in
# #   "${CMAKE_CURRENT_BINARY_DIR}/blob_detector/BlobDetectorConfig.cmake"
# #   @ONLY)

# # Create the BlobDetectorConfig.cmake and BlobDetectorConfigVersion files
# file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
#    "${INSTALL_INCLUDE_DIR}")

# # ... for the build tree
# set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
# configure_file(cmake/BlobDetectorConfig.cmake.in
#   "${PROJECT_BINARY_DIR}/BlobDetectorConfig.cmake" @ONLY)

# # ... for the install tree
# set(CONF_INCLUDE_DIRS "\${FOOBAR_CMAKE_DIR}/${REL_INCLUDE_DIR}")
# configure_file(cmake/BlobDetectorConfig.cmake.in
#   "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/BlobDetectorConfig.cmake" @ONLY)


# set(ConfigPackageLocation lib/cmake/blob_detector)

# message("CMAKE DIR: ${INSTALL_CMAKE_DIR}")

# # Install the ..Config.cmake and ..ConfigVersion.cmake
# install(FILES
#   "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/BlobDetectorConfig.cmake"
#   "${PROJECT_BINARY_DIR}/BlobDetectorConfigVersion.cmake"
#   DESTINATION "${INSTALL_CMAKE_DIR}"
#   # COMPONENT dev
# )

# # Install the export set for use with the install-tree
# install(EXPORT BlobDetectorTargets
# 	DESTINATION "${INSTALL_CMAKE_DIR}"
#   # COMPONENT dev
# )


# export(EXPORT blob_detector_targets
#   FILE "${CMAKE_CURRENT_BINARY_DIR}/blob_detector/blob_detector_targets.cmake"
# )

# configure_file(cmake/blob_detector_config.cmake
#   "${CMAKE_CURRENT_BINARY_DIR}/blob_detector/blob_detector_config.cmake"
#   COPYONLY
# )
