# EvolvingMemory

A high-performance CUDA library for evolving data buffers using genetic algorithms on GPUs. EvolvingMemory leverages parallel bit mutation and fitness-based selection to iteratively optimize data towards a target state.

## Overview

EvolvingMemory implements a genetic algorithm entirely on the GPU, allowing massive parallelization of:
- Random bit mutations (thousands of bits in parallel)
- Fitness evaluation (CPU or GPU-based)
- Memory operations optimized for 32-bit word access

The library supports both **CPU-based** and **GPU-based** fitness evaluation, with GPU kernels offering superior performance by avoiding device-to-host memory transfers.

## Features

- **Parallel Bit Mutation**: Mutate thousands of bits simultaneously using CUDA
- **Flexible Fitness Evaluation**: Choose between CPU functions or GPU kernels
- **32-bit Optimized**: Memory operations use 32-bit words for maximum throughput
- **Automatic Rollback**: Failed mutations are automatically reverted
- **Simple API**: Easy-to-use C++ interface
- **Header-only**: Single header and implementation file

## How It Works

1. **Initialize** random data buffer on GPU
2. **Backup** current state
3. **Mutate** random bits in parallel
4. **Evaluate** fitness (lower = better, 0.0 = perfect match)
5. **Restore** backup if fitness regressed, otherwise keep changes
6. **Repeat** until target found or max iterations reached

## Building

### Prerequisites

- CUDA Toolkit (11.0+)
- CMake (3.18+)
- C++11 compatible compiler
- NVIDIA GPU with compute capability 6.1+ (adjust in CMakeLists.txt)

### Build Instructions

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### CMake Options

- `BUILD_EXAMPLES` (default: ON) - Build example programs

### CUDA Architecture

By default, the library targets compute capabilities 6.1, 7.5, and 8.6. Adjust in [CMakeLists.txt:19](CMakeLists.txt#L19):

```cmake
CUDA_ARCHITECTURES "61;75;86"
```

## Usage

### CPU-Based Fitness Function

```cpp
#include "evolving_memory.cuh"

// Define a fitness function (lower is better)
float myFitnessFunction(const unsigned char* data, int dataSize, void* userData) {
    // Calculate how far data is from target
    // Return 0.0 for perfect match
    return calculateDistance(data, target);
}

int main() {
    // Configure
    EvolvingMemoryConfig config = {0};
    config.dataSize = 11;                      // Size of data buffer
    config.mutationRate = 0.1f;                // 10% of bits mutate per iteration
    config.maxIterations = 10000;
    config.fitnessFunc = myFitnessFunction;    // CPU function
    config.userData = targetData;              // Host memory pointer

    // Initialize
    EvolvingMemoryContext* ctx = evolvingMemoryInit(config);

    // Evolve
    int iterations = evolvingMemoryEvolve(ctx, config.maxIterations, true);

    // Retrieve result
    unsigned char* result = new unsigned char[config.dataSize];
    evolvingMemoryGetData(ctx, result);

    // Cleanup
    evolvingMemoryFree(ctx);
    delete[] result;
}
```

### GPU-Based Fitness Kernel (Recommended)

```cpp
// Define a GPU fitness kernel (no host transfers!)
void myFitnessKernel(const unsigned char* d_data, float* d_result,
                     int dataSize, void* d_userData) {
    // Launch CUDA kernels to compute fitness
    // Store result in d_result
    computeFitnessGPU<<<grid, block>>>(d_data, d_result, dataSize);
    cudaDeviceSynchronize();
}

int main() {
    // Allocate target on GPU
    unsigned char* d_target;
    cudaMalloc(&d_target, dataSize);
    cudaMemcpy(d_target, targetData, dataSize, cudaMemcpyHostToDevice);

    // Configure with GPU kernel
    EvolvingMemoryConfig config = {0};
    config.dataSize = 11;
    config.mutationRate = 0.1f;
    config.maxIterations = 10000;
    config.fitnessKernel = myFitnessKernel;    // GPU kernel
    config.d_userData = d_target;              // Device memory pointer

    // Initialize and evolve
    EvolvingMemoryContext* ctx = evolvingMemoryInit(config);
    evolvingMemoryEvolve(ctx, config.maxIterations, true);

    // Cleanup
    evolvingMemoryFree(ctx);
    cudaFree(d_target);
}
```

## Examples

### Hello World (CPU Fitness)

Evolves random data to match "HELLO WORLD" using CPU-based Hamming distance:

```bash
./build/hello_world
```

See [examples/hello_world.cu](examples/hello_world.cu) for the complete implementation.

### Hello World GPU (GPU Fitness)

Same example using GPU-based fitness kernel for better performance:

```bash
./build/hello_world_gpu
```

See [examples/hello_world_gpu.cu](examples/hello_world_gpu.cu) for the complete implementation.

## API Reference

### Configuration

```cpp
struct EvolvingMemoryConfig {
    int dataSize;                   // Size of data buffer in bytes
    float mutationRate;             // Mutation rate (0.0 to 1.0)
    int maxIterations;              // Maximum evolution iterations

    // Provide ONE of these:
    FitnessFunction fitnessFunc;    // CPU-based fitness function
    FitnessKernel fitnessKernel;    // GPU-based fitness kernel (faster)

    void* userData;                 // User data for CPU function (optional)
    void* d_userData;               // Device memory for GPU kernel (optional)
};
```

### Core Functions

#### `evolvingMemoryInit`
```cpp
EvolvingMemoryContext* evolvingMemoryInit(const EvolvingMemoryConfig& config);
```
Initialize evolving memory context. Returns NULL on error.

#### `evolvingMemoryEvolve`
```cpp
int evolvingMemoryEvolve(EvolvingMemoryContext* ctx, int maxIterations, bool verbose);
```
Run evolution loop. Returns number of iterations if successful, -1 if target not found.

#### `evolvingMemoryEvolveOnce`
```cpp
float evolvingMemoryEvolveOnce(EvolvingMemoryContext* ctx, int iteration);
```
Perform single evolution iteration. Returns new fitness score.

#### `evolvingMemoryGetData`
```cpp
void evolvingMemoryGetData(EvolvingMemoryContext* ctx, unsigned char* output);
```
Copy current data state to host memory.

#### `evolvingMemoryPrintState`
```cpp
void evolvingMemoryPrintState(EvolvingMemoryContext* ctx, int iteration);
```
Print current state and fitness to stdout.

#### `evolvingMemoryFree`
```cpp
void evolvingMemoryFree(EvolvingMemoryContext* ctx);
```
Free all resources and destroy context.

## Performance Tips

1. **Use GPU Fitness Kernels**: Avoid device-to-host transfers by implementing fitness evaluation entirely on GPU
2. **Tune Mutation Rate**: Start with 0.1 (10%) and adjust based on convergence speed
3. **Batch Operations**: The library automatically batches mutations for maximum parallelism
4. **CUDA Architecture**: Target your specific GPU architecture for optimal performance

## Implementation Details

### Parallel Mutation

The library mutates multiple bits simultaneously:
- Calculate number of bits to mutate: `dataSize * 8 * mutationRate`
- Launch one thread per bit mutation
- Each thread uses cuRAND for random bit selection
- Atomic operations prevent race conditions

### Memory Optimization

All kernels operate on 32-bit words when possible:
- **4x fewer memory operations** for aligned data
- Remaining bytes handled separately
- Coalesced memory access patterns

### Fitness Evaluation Modes

**CPU Mode**:
- Copies data from device to host
- Calls C++ fitness function
- Simple but slower due to PCIe transfers

**GPU Mode**:
- Evaluates fitness entirely on GPU
- No device-to-host transfers
- Significantly faster for large data

## License

This project is open source. Feel free to use and modify as needed.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional optimization algorithms
- Multi-population genetic algorithms
- Crossover operations
- Adaptive mutation rates
- Performance benchmarks

## Acknowledgments

Built with CUDA for high-performance parallel computing on NVIDIA GPUs.
