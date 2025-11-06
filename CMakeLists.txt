 # CMake minimum version
cmake_minimum_required(VERSION 3.10)

# Set project name and version
project(ElectronsAndProtons VERSION 1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Set CUDA properties
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optionally set build type to Release or Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

## Path to external libraries
#set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
#include_directories(${EXTERNAL_DIR}/glfw/include)

# Add include directories for CUDA
include_directories(${CUDA_INCLUDE_DIRS})

# GLFW (ensure it builds automatically)
#add_subdirectory(${EXTERNAL_DIR}/glfw)

include(FetchContent)
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4 # Use the specific version you want (e.g., latest stable tag)
)
FetchContent_MakeAvailable(glfw)

# Add executables for projects
add_executable(ElectronsAndProtons_CPU ${CMAKE_SOURCE_DIR}/ElectronsAndProtons_CPU/main.cpp)
add_executable(ElectronsAndProtons_CUDA ${CMAKE_SOURCE_DIR}/ElectronsAndProtons_CUDA/main.cu)

# Link libraries
target_link_libraries(ElectronsAndProtons_CPU PRIVATE
    glfw
    OpenGL::GL
)
target_link_libraries(ElectronsAndProtons_CUDA PRIVATE
    glfw
    OpenGL::GL
    CUDA::cudart
)

# For Linux, link to pthread
if(UNIX)
    target_link_libraries(ElectronsAndProtons_CPU PRIVATE pthread)
endif()

# To make sure CMake picks up OpenGL on Windows
if(WIN32)
    find_package(OpenGL REQUIRED)
    target_link_libraries(ElectronsAndProtons_CPU PRIVATE OpenGL::GL)
endif()

# Optimization
if (MSVC)
    add_compile_options(
        "$<$<CONFIG:Release>:/O3;/Oi;/Ot;/GL;/fp:fast;/arch:AVX2>"
        "$<$<CONFIG:RelWithDebInfo>:/O2;/fp:fast>"
    )
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        "$<$<CONFIG:Release>:-O3;-march=native;-ffast-math;-funroll-loops;-flto>"
        "$<$<CONFIG:RelWithDebInfo>:-O3;-ffast-math>"
    )
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS_RELEASE}")