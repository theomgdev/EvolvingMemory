cmake_minimum_required(VERSION 3.18)
project(EvolvingMemory CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Create the library
add_library(evolving_memory_lib STATIC
    evolving_memory.cu
    evolving_memory.cuh
)

# Set CUDA architectures (adjust based on your GPU)
set_target_properties(evolving_memory_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61;75;86"
)

# Link CUDA libraries to the library
target_link_libraries(evolving_memory_lib ${CUDA_LIBRARIES} curand)

# Include directory for the library
target_include_directories(evolving_memory_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Build examples
option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_EXAMPLES)
    # Hello World example
    add_executable(hello_world examples/hello_world.cu)
    set_target_properties(hello_world PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "61;75;86"
    )
    target_link_libraries(hello_world evolving_memory_lib)
endif()
