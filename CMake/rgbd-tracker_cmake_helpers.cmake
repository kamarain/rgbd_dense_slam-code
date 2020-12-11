# This files probe directories defined by user "-D<LIB_INSTALL_DIR>"
# and set <LIBNAME>_LIBRARIES variable (note that this can be different
# if using FIND_PACKAGE (in that case just add also definition of that
# variable)


# Tries to guess GLEW include and lib paths and builds own
FUNCTION(CheckLocalGLEW)
  MESSAGE(STATUS "Checking file: ${GLEW_DIR_SAVE}include/GL/glew.h")
  IF(EXISTS "${GLEW_DIR_SAVE}/include/GL/glew.h")
    MESSAGE(STATUS "FIND_PACKAGE failed with GLEW, but a local installation found by -DGLEW_DIR option - using this")
    INCLUDE_DIRECTORIES(${GLEW_DIR_SAVE}/include)
    # For some reason stopped working
    #ADD_LIBRARY(glew STATIC IMPORTED)
    #SET_TARGET_PROPERTIES(glew PROPERTIES
    #  IMPORTED_LOCATION ${GLEW_DIR_SAVE}/lib/libGLEW.a)
    #SET(GLEW_LIBRARY glew) 
    SET(GLEW_LIBRARIES "${GLEW_DIR_SAVE}/lib/libGLEW.a" PARENT_SCOPE)
    SET(GLEW_LIBS "${GLEW_DIR_SAVE}/lib/libGLEW.a" PARENT_SCOPE)
    MESSAGE(STATUS "GLEW_LIBRARIES SET TO ${GLEW_DIR_SAVE}/lib/libGLEW.a")
  ELSE(EXISTS "${GLEW_DIR_SAVE}/include/GL/glew.h")
    MESSAGE(FATAL_ERROR "This project really needs GLEW but cannot find!")
  ENDIF(EXISTS "${GLEW_DIR_SAVE}/include/GL/glew.h")
ENDFUNCTION()

# Tries to guess TinyXML include and lib paths and builds own
FUNCTION(CheckLocalTinyXML)
  MESSAGE(STATUS "Checking file: ${TINYXML_DIR_SAVE}tinyxml.h")
  IF(EXISTS "${TINYXML_DIR_SAVE}tinyxml.h")
    MESSAGE(STATUS "FIND_PACKAGE failed with tinyXML, but a local installation found by -DTINYXML_DIR option - using this")
    INCLUDE_DIRECTORIES(${TINYXML_DIR_SAVE})
    # Since TinyXML is only object files, we need to do some tricks:
    
    SET(TINYXML_SOURCES
      ${TINYXML_DIR_SAVE}tinyxml.h ${TINYXML_DIR_SAVE}tinyxml.cpp
      ${TINYXML_DIR_SAVE}tinystr.h ${TINYXML_DIR_SAVE}tinystr.cpp
      ${TINYXML_DIR_SAVE}tinyxmlerror.cpp
      ${TINYXML_DIR_SAVE}tinyxmlparser.cpp)
    ADD_LIBRARY(tinyxml STATIC
      ${TINYXML_SOURCES})
    SET(TINYXML_LIBRARY tinyxml PARENT_SCOPE) 
    SET(TINYXML_LIBRARIES tinyxml PARENT_SCOPE) 
    SET(TINYXML_LIBS tinyxml PARENT_SCOPE) 
  ELSE(EXISTS "${TINYXML_DIR_SAVE}tinyxml.h")
    MESSAGE(FATAL_ERROR "This project really needs TinyXML but cannot find!")
  ENDIF(EXISTS "${TINYXML_DIR_SAVE}tinyxml.h")
ENDFUNCTION()

FUNCTION(CheckLocalFreeNect)
  MESSAGE(STATUS "Checking file: ${FREENECT_DIR_SAVE}include/libfreenect.h")
  IF(EXISTS "${FREENECT_DIR_SAVE}/include/libfreenect.h")
    MESSAGE(STATUS "FIND_PACKAGE failed with FreeNect, but a local installation found by -DFreeNect_DIR option - using this")
    INCLUDE_DIRECTORIES(${FREENECT_DIR_SAVE}/include)
    # For some reason stopped working
    #ADD_LIBRARY(freenect STATIC IMPORTED)
    #SET(FREENECT_LIBRARY freenect) 
    #SET_TARGET_PROPERTIES(freenect PROPERTIES
    #  IMPORTED_LOCATION ${FREENECT_DIR_SAVE}build/lib/libfreenect.a)
    SET(FREENECT_LIBRARIES "${FREENECT_DIR_SAVE}build/lib/libfreenect.a;${FREENECT_DIR_SAVE}build/lib/libfreenect_sync.a" PARENT_SCOPE) 
    SET(FREENECT_LIBS "${FREENECT_DIR_SAVE}build/lib/libfreenect.a;${FREENECT_DIR_SAVE}build/lib/libfreenect_sync.a" PARENT_SCOPE) 
    MESSAGE(STATUS "FREENECT_LIBRARIES SET TO ${FREENECT_DIR_SAVE}build/lib/libfreenect.a;${FREENECT_DIR_SAVE}build/lib/libfreenect_sync.a")
  ELSE(EXISTS "${FREENECT_DIR_SAVE}/include/libfreenect.h")
    MESSAGE(FATAL_ERROR "This project really needs FreeNect but cannot find!")
  ENDIF(EXISTS "${FREENECT_DIR_SAVE}/include/libfreenect.h")
ENDFUNCTION()