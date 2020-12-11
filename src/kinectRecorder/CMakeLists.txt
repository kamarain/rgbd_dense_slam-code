CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# This is maybe wrong as these all should be in public "include" of the gputracker library
INCLUDE_DIRECTORIES(
  "include"
  ${CMAKE_PUBLIC_HEADERS_OUTPUT_DIRECTORY}/
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/gputracker/
  ${CMAKE_SOURCE_DIR}/src/gputracker/cudakernels/
  ${CMAKE_SOURCE_DIR}/src/gputracker/utils/
  ${CMAKE_SOURCE_DIR}/src/gputracker/utils/capture/
  ${CMAKE_SOURCE_DIR}/src/gputracker/utils/multicore/
  ${CMAKE_SOURCE_DIR}/src/gputracker/utils/reconstruct/
  )

SET(KINECTRECORDER_SOURCES
  kinectRecorder.cpp
  testApplication.cpp
  )

SET(KINECTRECORDER_HEADERS
  testApplication.h
  )

CUDA_ADD_EXECUTABLE(kinectRecorder ${KINECTRECORDER_SOURCES} ${KINECTRECORDER_HEADERS})

# Note that the correct order of libs matters (based on dependencies)
TARGET_LINK_LIBRARIES(kinectRecorder ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${X11_X11_LIB} ${SDL_LIBRARY} ${FREETYPE_LIBRARY} 
${OpenCV_LIBS} gputracker tinyxml PoissonRecon ${CUDA_LIBRARIES} ${GLEW_LIBRARY} ${FREENECT_LIBRARIES} ${LIBUSB_1_LIBRARIES} png12)

