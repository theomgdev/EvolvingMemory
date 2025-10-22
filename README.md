# Evolving Memory

A CUDA-based genetic algorithm library that evolves random binary data towards a target string using parallel bit mutations.

## Concept

The algorithm works as follows:

1. **Initialize**: Start with random bytes in GPU memory
2. **Mutate**: Flip random bits in parallel on the GPU
3. **Evaluate**: Calculate fitness (Hamming distance to target)
4. **Select**: Keep changes if fitness improved or stayed same, otherwise revert
5. **Repeat**: Continue until target is found or max iterations reached

## Features

- Library-based API for easy integration
- Parallel bit mutations using CUDA
- Configurable mutation rate and data size
- Simple fitness function (Hamming distance)
- Automatic rollback on regression
- Real-time progress display

## Library Usage

### Include the library

```cpp
#include "evolving_memory.cuh"
```

### Basic Example

```cpp
// Configure evolving memory
EvolvingMemoryConfig config;
config.dataSize = 11;              // "HELLO WORLD" is 11 characters
config.mutationRate = 0.1f;        // 10% mutation rate
config.maxIterations = 10000;      // Maximum iterations
config.target = "HELLO WORLD";     // Target string

// Initialize evolving memory
EvolvingMemoryContext* ctx = evolvingMemoryInit(config);
if (!ctx) {
    fprintf(stderr, "Failed to initialize evolving memory\n");
    return 1;
}

// Run evolution with verbose output
int iterations = evolvingMemoryEvolve(ctx, true);

// Get final data
unsigned char finalData[11];
evolvingMemoryGetData(ctx, finalData);

// Cleanup
evolvingMemoryFree(ctx);
```

### API Functions

- `evolvingMemoryInit(config)` - Initialize context with configuration
- `evolvingMemoryFree(ctx)` - Free context and GPU memory
- `evolvingMemoryEvolve(ctx, verbose)` - Run full evolution loop
- `evolvingMemoryEvolveOnce(ctx, iteration)` - Run single iteration
- `evolvingMemoryGetData(ctx, output)` - Get current data state
- `evolvingMemoryPrintState(ctx, iteration)` - Print current state
- `evolvingMemoryCalculateFitness(ctx)` - Calculate fitness score

## Build Instructions

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- CMake 3.18 or higher
- C++ compiler

### Build Library and Examples

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Build Library Only (No Examples)

```bash
cmake -DBUILD_EXAMPLES=OFF ..
cmake --build .
```

### Run Examples

```bash
# Hello World example
./hello_world
```

## Example Output

```
=== Evolving Memory - Simple MVP ===
Target: "HELLO WORLD"
Data Size: 11 bytes
Mutation Rate: 10.0%
=====================================

Initial state:
Iteration 0: "a7#x9$@..." | Fitness: 45

Starting evolution...

Iteration 100: "HE..O W...." | Fitness: 28
Iteration 250: "HELLO W...." | Fitness: 12
Iteration 389: "HELLO WORLD" | Fitness: 0

*** SUCCESS! Found target in 389 iterations ***
```

## How It Works

### 1. Parallel Mutation (GPU)
The `mutateBits` kernel runs in parallel, with each thread responsible for flipping one random bit:
- Backs up current data
- Uses cuRAND for random bit selection
- Atomically flips bits to avoid race conditions

### 2. Fitness Evaluation (CPU)
Simple Hamming distance calculation:
- XOR each byte with target byte
- Count differing bits
- Lower score = better fitness

### 3. Selection Strategy
- If `new_fitness <= previous_fitness`: keep changes
- If `new_fitness > previous_fitness`: restore backup using GPU

## Future Improvements

- Multiple parallel populations
- Adaptive mutation rate
- Different fitness functions
- Crossover operations
- GPU-based fitness calculation
- Performance metrics and benchmarking

## License

MIT
