cmake_minimum_required (VERSION 2.6)

IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

 # should we use our own math functions?
option (VERBOSE_MODE
	 "Verbose mode" OFF) 

SET(CMAKE_FIND_LIBRARY_PREFIXES "lib" "")
SET(CMAKE_FIND_LIBRARY_SUFFIXES "d" "")

project (stereogen)
# The version number.
set (STEREOGEN_VERSION_MAJOR 1)
set (STEREOGEN_VERSION_MINOR 0)
  
# add the binary tree to the search path for include files
# so that we will find SDLconfig.h
include_directories("${PROJECT_BINARY_DIR}")

set(CMAKE_CXX_FLAGS_RELEASE "-m64 -O3 -fopenmp -Wno-unused-result -Wno-write-strings")
set(CMAKE_C_FLAGS_RELEASE "-m64 -O3 -fopenmp -Wno-unused-result -Wno-write-strings")
set(CMAKE_CXX_FLAGS_DEBUG "-m64 -g -O0 -fopenmp -Wno-unused-result -Wno-write-strings")
set(CMAKE_C_FLAGS_DEBUG "-m64 -g -O0 -fopenmp -Wno-unused-result -Wno-write-strings")
set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_LIBRARY_POSTFIX "")
# Default places for archives (static), libs (static and shared) and binaries
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
if (VERBOSE_MODE)
	set(CMAKE_VERBOSE_MAKEFILE ON)
endif (VERBOSE_MODE)

include_directories ("${PROJECT_SOURCE_DIR}")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
find_package(CUDA 5.5 REQUIRED)
set(CUDA_INCLUDE_DIRS "${CUDA_INCLUDE_DIRS},${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc")
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
# turn on CUDA opt
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -Xcompiler -m64 -arch=sm_30")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc")
FIND_PACKAGE(OpenCV COMPONENTS highgui nonfree features2d imgproc flann core video calib3d REQUIRED)
FIND_PACKAGE(GLEW)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(X11)
FIND_PACKAGE(GLUT REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
# Enable this to compile the library
ADD_SUBDIRECTORY(ext)
add_subdirectory(stereogen)







