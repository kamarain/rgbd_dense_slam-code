if(EXISTS "${CMAKE_SOURCE_DIR}/ext/poisson/PoissonRecon/Src/PoissonRecon.cpp")
  message(STATUS "PoissonRecon source files exist, good for you!")
  if(EXISTS "${CMAKE_PUBLIC_HEADERS_OUTPUT_DIRECTORY}/PoissonRecon")
  
  else()
    message(STATUS "Copying PoissonRecon headers to ${CMAKE_PUBLIC_HEADERS_OUTPUT_DIRECTORY}/PoissonRecon/")
    file( COPY PoissonRecon/Src/
      DESTINATION ${CMAKE_PUBLIC_HEADERS_OUTPUT_DIRECTORY}/PoissonRecon/
      PATTERN "*.cpp" EXCLUDE 
      PATTERN "*.tc" EXCLUDE 
      PATTERN "*.c" EXCLUDE)
    file( COPY ./PoissonRecon-wrapper.h
      DESTINATION ${CMAKE_PUBLIC_HEADERS_OUTPUT_DIRECTORY}/PoissonRecon/)
  endif()
else()
  message(SEND_ERROR "PoissonRecon source files not found! Run source get_PoissonRecon.sh within src/external/ directory")
endif()

INCLUDE_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}/ext/poisson/
  ${CMAKE_SOURCE_DIR}/ext/poisson/PoissonRecon/Src/
  )

# PoissonRecon
SET(PoissonRecon_SOURCES
  PoissonRecon/Src/CmdLineParser.cpp
  PoissonRecon/Src/Factor.cpp
  PoissonRecon/Src/Geometry.cpp
  PoissonRecon/Src/MarchingCubes.cpp
  PoissonRecon/Src/PlyFile.cpp
  PoissonRecon/Src/Time.cpp
  PoissonRecon-wrapper.cpp
  )

#
SET(PoissonRecon_HEADERS
  PoissonRecon/Src/CmdLineParser.h
  PoissonRecon/Src/Factor.h
  PoissonRecon/Src/Geometry.h
  PoissonRecon/Src/MarchingCubes.h
  PoissonRecon/Src/PlyFile.h
  PoissonRecon/Src/Time.h
  PoissonRecon-wrapper.h
)

ADD_LIBRARY(PoissonRecon
  ${PoissonRecon_SOURCES} ${PoissonRecon_HEADERS})
