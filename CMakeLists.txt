cmake_minimum_required(VERSION 3.18)
project(EvolvingMemory CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Add executable
add_executable(evolving_memory main.cu)

# Set CUDA architectures (adjust based on your GPU)
set_target_properties(evolving_memory PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61;75;86"
)

# Link CUDA libraries
target_link_libraries(evolving_memory ${CUDA_LIBRARIES} curand)
