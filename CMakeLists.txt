cmake_minimum_required(VERSION 3.22)

project(ppe CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

# Find CUDA package

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenMP package
find_package(OpenMP)
# Find the Boost package
find_package(Boost COMPONENTS system REQUIRED)

# Add C++ source files
set(CPP_SOURCES
        custom_types.cpp
        dct8x8_block.cpp
        xml_aux.cpp
        )

# Add CUDA source file
set(CUDA_SOURCES
        main.cu
        )

# Specify the target executable
add_executable(ppe ${CPP_SOURCES} ${CUDA_SOURCES})

# Set CUDA architecture (adjust based on your target GPU)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_35)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx512f")

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)

# Link CUDA libraries
target_link_libraries(ppe xml2 tiff cudart OpenMP::OpenMP_CXX tcmalloc -march=native -mavx512f ${Boost_SYSTEM_LIBRARY})

# Optionally, set compiler flags or include directories if needed
# target_compile_options(test-cuda PRIVATE ...)

# Optionally, add include directories if needed
target_include_directories(ppe PRIVATE -I /usr/include/libxml2/ ${Boost_INCLUDE_DIRS})

# Optionally, set compiler flags for specific files
set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS "-Xcompiler -fopenmp -Xcompiler -mavx512f")

# Optionally, set compiler flags for CUDA files
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE CXX)

set_target_properties(ppe PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)




