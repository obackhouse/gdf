cmake_minimum_required(VERSION 3.5)
project("gdf")

# OpenMP
set(OPENMP_C_PROPERTIES "")
find_package(OpenMP)
if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
else()
    set(OpenMP_C_FLAGS " ")
endif()

# BLAS
if (NOT BLAS_LIBRARIES)
    find_package(BLAS)
endif()
if (APPLE)
    include_directories("${BLAS_LIBRARIES}/Frameworks/vecLib.framework/Headers")
endif()
if (NOT BLAS_LIBRARIES)
    message(FATAL_ERROR "A required library with BLAS API not found.")
else()
    message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()

# RPATH
if (APPLE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
    set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
else()
    set(CMAKE_SKIP_BUILD_RPATH  True)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
endif()

# C flags
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")
include(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# Make the gdf library and add gdf/lib/src/jk.c
add_library(gdf SHARED src/jk.c)
target_include_directories(gdf PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(gdf PRIVATE ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
target_compile_options(gdf PRIVATE ${OPENMP_C_FLAGS})

# Move gdf/lib.cpython-37m-x86_64-linux-gnu.so to gdf/lib/libgdf.so
set_target_properties(gdf PROPERTIES PREFIX "")
set_target_properties(gdf PROPERTIES OUTPUT_NAME "gdf")
set_target_properties(gdf PROPERTIES SUFFIX ".so")
set_target_properties(gdf PROPERTIES LIBRARY_OUTPUT_DIRECTORY .)
