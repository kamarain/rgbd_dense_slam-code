CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# Set this to see full compilation command lines
SET(CMAKE_VERBOSE_MAKEFILE OFF)

#SET (CMAKE_CXX_COMPILER "/usr/bin/g++")

# Directory to additional cmake helpers
SET(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/CMake" ${CMAKE_MODULE_PATH})

# Project needs to be defined before FIND_PACKAGE
# since it sets the required environment variables
PROJECT(rgbd-tracker)

# Default places for archives (static), libs (static and shared) and
# binaries
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_PUBLIC_HEADERS_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/include)

# Find required packages
FIND_PACKAGE(GLEW) # not necessary cmake find helper available
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV COMPONENTS highgui nonfree features2d imgproc flann core video calib3d REQUIRED)
#FIND_PACKAGE(TinyXML) # not necessarily cmake find helper available
FIND_PACKAGE(FreeNect) # not necessarily cmake find helper available
FIND_PACKAGE(SDL REQUIRED)
FIND_PACKAGE(Freetype REQUIRED)
FIND_PACKAGE(PNG REQUIRED)
FIND_PACKAGE(libusb-1.0 REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)

# Add CUDA directories to the path
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc")
INCLUDE_DIRECTORIES("${CUDA_TOOLKIT_ROOT_DIR}/samples/3_Imaging/histogram")
INCLUDE_DIRECTORIES("${CUDA_TOOLKIT_ROOT_DIR}/samples/6_Advanced/reduction")
INCLUDE_DIRECTORIES("${CUDA_TOOLKIT_ROOT_DIR}/samples/5_Simulations/smokeParticles")
INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIR_freetype2})
INCLUDE_DIRECTORIES(${LIBUSB_1_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${FREENECT_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("${CMAKE_HOME_DIRECTORY}/ext/tinyxml")
INCLUDE_DIRECTORIES(${GLEW_INCLUDE_PATH})
# Adjusting nvcc flags
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -Xcompiler -m64 -arch=sm_20")

# CXX compilation flags
#SET(CMAKE_CXX_FLAGS "-m64 -O3 -DGLEW_STATIC -fopenmp -Wno-unused-result -Wno-write-strings -DNO_SDL_GLEXT")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result -Wno-write-strings -DGLEW_STATIC -m64 -O3 -DNO_SDL_GLEXT")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -Wno-deprecated") # For PoissonRecon

# Enable this to compile the library
ADD_SUBDIRECTORY(ext)
ADD_SUBDIRECTORY(src)
