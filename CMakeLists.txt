# CMake minimum version
cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW) # For CUDA architectures

# Set project name and version
project(ElectronsAndProtons VERSION 1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set CUDA properties
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# --- portable CUDA architecture selection ---
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (CUDAToolkit_VERSION VERSION_LESS 11.0)
        # Very old toolkit (Kepler through Turing)
        set(CMAKE_CUDA_ARCHITECTURES "30-real;35-real;50-real;60-real;70-real;75-real")
    elseif (CUDAToolkit_VERSION VERSION_LESS 12.0)
        # CUDA 11.x (Kepler still supported)
        set(CMAKE_CUDA_ARCHITECTURES "35-real;50-real;60-real;70-real;75-real;86-real;89-real")
    else()
        # CUDA 12+ (Kepler dropped, Hopper introduced)
        set(CMAKE_CUDA_ARCHITECTURES "50-real;60-real;70-real;75-real;86-real;89-real;90-virtual")
    endif()

    message(STATUS "Detected CUDA Toolkit ${CUDAToolkit_VERSION}, using architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Add include directories for CUDA
include_directories(${CUDA_INCLUDE_DIRS})

# Set build type to Release if build type not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/shared/include)

# Fetch GLFW from GitHub
include(FetchContent)
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4 # Use the specific version you want (e.g., latest stable tag)
)
FetchContent_MakeAvailable(glfw)

# OpenGL (cross-platform)
find_package(OpenGL REQUIRED)

# Wszystkie wspólne pliki źródłowe w shared/src
file(GLOB SHARED_SRC
    "${CMAKE_SOURCE_DIR}/shared/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/shared/src/*.cu"
)

# CPU
set(CPU_SRC
    "${CMAKE_SOURCE_DIR}/ElectronsAndProtons_CPU/main.cpp"
)

# CUDA
set(CUDA_SRC
    "${CMAKE_SOURCE_DIR}/ElectronsAndProtons_CUDA/main.cu"
)

# Add executables for projects
add_executable(ElectronsAndProtons_CPU
    ${CPU_SRC}
    ${SHARED_SRC}
)

add_executable(ElectronsAndProtons_CUDA
    ${CUDA_SRC}
    ${SHARED_SRC}
)

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
    target_link_libraries(ElectronsAndProtons_CPU PRIVATE OpenGL::GL)
endif()

# Optimization
if (MSVC)
    add_compile_options(
        "$<$<CONFIG:Release>:/O2;/Oi;/Ot;/GL;/fp:fast;/arch:AVX2>"
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
