# - Try to find freenect
# Once done this will define
#  FREENECT_FOUND
#  FREENECT_INCLUDE_DIRS
#  FREENECT_LIBRARIES
#  FREENECT_DEFINITIONS - Compiler switches required

find_path(FREENECT_INCLUDE_DIR
  NAMES libfreenect.h
  PATHS /usr/local/include/libfreenect)

find_library(FREENECT_LIBRARY
  NAMES libfreenect.a
  PATHS /usr/local/lib64)

set(FREENECT_LIBRARIES ${FREENECT_LIBRARY} )
set(FREENECT_INCLUDE_DIRS ${FREENECT_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set <module>_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(freenect
  DEFAULT_MSG
  FREENECT_LIBRARY
FREENECT_INCLUDE_DIR)

mark_as_advanced(FREENECT_INCLUDE_DIR FREENECT_LIBRARY)
