# FindGMP.cmake
# Created by Kiro
# Find the GNU Multiple Precision Arithmetic Library (GMP)
#
# This module defines:
#  GMP_FOUND - True if GMP is found
#  GMP_INCLUDE_DIRS - Include directories for GMP
#  GMP_LIBRARIES - Libraries to link against for GMP
#  GMP_VERSION - Version of GMP found
#
# This module also creates the imported target:
#  GMP::GMP - The GMP library target

# Find the header file
find_path(GMP_INCLUDE_DIR
    NAMES gmp.h
    HINTS
        ${GMP_ROOT}
        $ENV{GMP_ROOT}
        ${GMP_PREFIX}
        $ENV{GMP_PREFIX}
    PATH_SUFFIXES include
    DOC "GMP include directory"
)

# Find the library
find_library(GMP_LIBRARY
    NAMES gmp libgmp
    HINTS
        ${GMP_ROOT}
        $ENV{GMP_ROOT}
        ${GMP_PREFIX}
        $ENV{GMP_PREFIX}
    PATH_SUFFIXES lib lib64
    DOC "GMP library"
)

# Try to extract version information from gmp.h
if(GMP_INCLUDE_DIR AND EXISTS "${GMP_INCLUDE_DIR}/gmp.h")
    file(STRINGS "${GMP_INCLUDE_DIR}/gmp.h" GMP_VERSION_MAJOR_LINE
         REGEX "^#define[ \t]+__GNU_MP_VERSION[ \t]+[0-9]+")
    file(STRINGS "${GMP_INCLUDE_DIR}/gmp.h" GMP_VERSION_MINOR_LINE
         REGEX "^#define[ \t]+__GNU_MP_VERSION_MINOR[ \t]+[0-9]+")
    file(STRINGS "${GMP_INCLUDE_DIR}/gmp.h" GMP_VERSION_PATCH_LINE
         REGEX "^#define[ \t]+__GNU_MP_VERSION_PATCHLEVEL[ \t]+[0-9]+")
    
    string(REGEX REPLACE "^#define[ \t]+__GNU_MP_VERSION[ \t]+([0-9]+).*" "\\1"
           GMP_VERSION_MAJOR "${GMP_VERSION_MAJOR_LINE}")
    string(REGEX REPLACE "^#define[ \t]+__GNU_MP_VERSION_MINOR[ \t]+([0-9]+).*" "\\1"
           GMP_VERSION_MINOR "${GMP_VERSION_MINOR_LINE}")
    string(REGEX REPLACE "^#define[ \t]+__GNU_MP_VERSION_PATCHLEVEL[ \t]+([0-9]+).*" "\\1"
           GMP_VERSION_PATCH "${GMP_VERSION_PATCH_LINE}")
    
    set(GMP_VERSION "${GMP_VERSION_MAJOR}.${GMP_VERSION_MINOR}.${GMP_VERSION_PATCH}")
endif()

# Handle the QUIETLY and REQUIRED arguments and set GMP_FOUND to TRUE if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP
    REQUIRED_VARS GMP_LIBRARY GMP_INCLUDE_DIR
    VERSION_VAR GMP_VERSION
)

if(GMP_FOUND)
    set(GMP_LIBRARIES ${GMP_LIBRARY})
    set(GMP_INCLUDE_DIRS ${GMP_INCLUDE_DIR})
    
    # Create imported target
    if(NOT TARGET GMP::GMP)
        add_library(GMP::GMP UNKNOWN IMPORTED)
        set_target_properties(GMP::GMP PROPERTIES
            IMPORTED_LOCATION "${GMP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}"
        )
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARY)