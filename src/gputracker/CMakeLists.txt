# This defines building gputracker lib for CUDA
# enchanced visual SLAM functionality

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# these should be fixed as some refer to <util1dir>/<util1file> and
# some <util1file> which requires adding it to include dirs
INCLUDE_DIRECTORIES(
  ${CMAKE_BINARY_DIR}/include/
  ${CMAKE_SOURCE_DIR}/src/gputracker/
  ${CMAKE_SOURCE_DIR}/src/gputracker/cudakernels/
  ${CMAKE_SOURCE_DIR}/src/gputracker/utils/
  ${CMAKE_SOURCE_DIR}/src/gputracker/utils/capture/
  ${CMAKE_SOURCE_DIR}/src/gputracker/utils/multicore/
  ${CMAKE_SOURCE_DIR}/src/gputracker/utils/reconstruct/
  )

# CUDA KERNELS
SET(CUDA_KERNEL_SOURCES
  #cudakernels/square_array.cu
  cudakernels/d2z.cu
  cudakernels/downSample2Cuda.cu
  cudakernels/hostUtils.cpp
  cudakernels/rgb2gray.cu
  cudakernels/undistort.cu
  cudakernels/convert2FloatCuda.cu
  cudakernels/adapted_histograms.cu
  cudakernels/warp.cu
  #cudakernels/matrixMul_kernel.cu
  #cudakernels/cuda_hist.cu
)

# GPUTRACKER
SET(GPUTRACKER_SOURCES
  gputracker.cpp
  )

SET(GPUTRACKER_HEADERS
  gputracker.h)

# CALIB
SET(CALIB_SOURCES
  utils/calib/calib.cpp
  utils/calib/GroundTruth.cpp
  utils/calib/ResultTxt.cpp
  )

SET(CALIB_HEADERS
  utils/calib/calib.h
  utils/calib/GroundTruth.h
  utils/calib/ResultTxt.h
  )

# CAMERA - not used (anymore?)
SET(CAMERA_SOURCES
  utils/camera/Camera.cpp
  utils/camera/InteractiveCamera.cpp
  utils/camera/Matrix.cpp
  utils/camera/OrbitingCamera.cpp
  #src/utils/camera/PerformanceTimer.cpp
  utils/camera/Quaternion.cpp
  utils/camera/Vector.cpp
  )

SET(CAMERA_HEADERS
  utils/camera/Camera.hpp
  utils/camera/commonmath.h
  utils/camera/LinkedList.h
  utils/camera/OrbitingCamera.h
  #src/utils/camera/PerformanceTimer.h
  utils/camera/InteractiveCamera.hpp
  utils/camera/Matrix.hpp
  utils/camera/Quaternion.hpp
  utils/camera/Vector.hpp
  )

# EXPM (expo toolkit for matrix exponentials)
SET(EXPM_SOURCES
  utils/expm/expmHost.cpp
  )

SET(EXPM_HEADERS
  utils/expm/expmHost.h
  )

# IMAGEMATCH (ICP tools)
SET(IMAGEMATCH_SOURCES
  utils/imagematch/ICP.cpp
  )

SET(IMAGEMATCH_HEADERS
  utils/imagematch/ICP.h
  )

# IMAGE2
SET(IMAGE2_SOURCES
  utils/image2/Image2.cpp
  utils/image2/ImagePyramid2.cpp
  )

SET(IMAGE2_HEADERS
  utils/image2/Image2.h
  utils/image2/ImagePyramid2.h
  )

# MULTICORE
SET(MULTICORE_SOURCES
  utils/multicore/multicore.cpp
  )

SET(MULTICORE_HEADERS
  utils/multicore/multicore.h
  )

# RENDERING
SET(RENDERING_SOURCES
  utils/rendering/BaseBuffer2.cpp
  utils/rendering/LineBuffer.cpp
  utils/rendering/shader.cpp
  utils/rendering/TriangleBuffer2.cpp
  utils/rendering/VertexBuffer2.cpp
  utils/rendering/teapot.cpp
  utils/rendering/GLWindow.cpp
)

SET(RENDERING_HEADERS
  utils/rendering/BaseBuffer2.h
  utils/rendering/LineBuffer.h
  utils/rendering/shader.h
  utils/rendering/TriangleBuffer2.h
  utils/rendering/VertexBuffer2.h
  utils/rendering/teapot.h
  utils/rendering/GLWindow.h
)

# STRUCTURE
SET(STRUCTURE_SOURCES
  utils/structure/configuration.cpp
)

SET(STRUCTURE_HEADERS
  utils/structure/configuration.h
)

# WARPING
SET(WARPING_SOURCES
  utils/warping/KeyFrameRing.cpp
#  src/utils/warping/posefilter.cpp
)

SET(WARPING_HEADERS
  utils/warping/KeyFrameRing.h
#  src/utils/warping/posefilter.h
)

# CAPTURE
SET(CAPTURE_SOURCES
  utils/capture/fileSource.cpp
  utils/capture/Kinect.cpp
  utils/capture/KinectDisparityCompressor.cpp
  utils/capture/VideoPreProcessorCPU.cpp
  utils/capture/VideoPreProcessorGPU.cpp
)

SET(CAPTURE_HEADERS
  utils/capture/fileSource.h
  utils/capture/kinect.h
  utils/capture/KinectDisparityCompressor.h
  utils/capture/VideoPreProcessorCPU.h
  utils/capture/VideoPreProcessorGPU.h
)

# RECONSTRUCT
SET(RECONSTRUCT_SOURCES
  utils/reconstruct/KeyFrame.cpp
  utils/reconstruct/KeyFrameModel.cpp
  utils/reconstruct/basic_math.cpp
  utils/reconstruct/zconv.cpp
)

SET(RECONSTRUCT_HEADERS
  utils/reconstruct/KeyFrame.h
  utils/reconstruct/KeyFrameModel.h
  utils/reconstruct/basic_math.h
  utils/reconstruct/zconv.h
)


CUDA_ADD_LIBRARY(gputracker
  ${GPUTRACKER_SOURCES} ${GPUTRACKER_HEADERS}
  ${CALIB_SOURCES} ${CALIB_HEADERS}
  ${CAMERA_SOURCES} ${CAMERA_HEADERS}
  ${CAPTURE_SOURCES} ${CAPTURE_HEADERS}
  ${EXPM_SOURCES} ${EXPM_HEADERS}
  ${IMAGE2_SOURCES} ${IMAGE2_HEADERS}
  ${IMAGEMATCH_SOURCES} ${IMAGEMATCH_HEADERS}
  ${MULTICORE_SOURCES} ${MULTICORE_HEADERS}
  ${RENDERING_SOURCES} ${RENDERING_HEADERS}
  ${STRUCTURE_SOURCES} ${STRUCTURE_HEADERS}
  ${WARPING_SOURCES} ${WARPING_HEADERS}
  ${CUDA_KERNEL_SOURCES}
  ${RECONSTRUCT_SOURCES} ${RECONSTRUCT_HEADERS})

