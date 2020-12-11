#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARY
#

find_path(GLEW_INCLUDE_PATH GL/glew.h 
	PATHS /usr/local/cuda-5.5/samples/common/inc /usr/local/include /usr/include
	DOC "The GLEW include path"
 	NO_SYSTEM_ENVIRONMENT_PATH
	NO_DEFAULT_PATH
)

FIND_LIBRARY( GLEW_LIBRARY
NAMES libGLEW.a
PATHS
/usr/local/cuda-5.5/samples/common/lib/linux/x86_64
/usr/lib64
/usr/lib
/usr/local/lib64
/usr/local/lib
/sw/lib
/opt/local/lib
DOC "The GLEW library")

IF (GLEW_INCLUDE_PATH)
SET( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ELSE (GLEW_INCLUDE_PATH)
SET( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (GLEW_INCLUDE_PATH)

MARK_AS_ADVANCED( GLEW_FOUND )
