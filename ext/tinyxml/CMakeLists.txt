set(TINYXML_PATH ${CMAKE_SOURCE_DIR}/ext/tinyxml )
INCLUDE_DIRECTORIES(${TINYXML_PATH})

# Since TinyXML is only object files, we need to do some tricks:
SET(TINYXML_SOURCES
      ${TINYXML_PATH}/tinyxml.h ${TINYXML_PATH}/tinyxml.cpp
      ${TINYXML_PATH}/tinystr.h ${TINYXML_PATH}/tinystr.cpp
      ${TINYXML_PATH}/tinyxmlerror.cpp
      ${TINYXML_PATH}/tinyxmlparser.cpp)
ADD_LIBRARY(tinyxml STATIC ${TINYXML_SOURCES})

