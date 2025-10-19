# Evolving Memory

A simple CUDA-based genetic algorithm that evolves random binary data towards a target string using parallel bit mutations.

## Concept

The algorithm works as follows:

1. **Initialize**: Start with random bytes in GPU memory (default: 11 bytes)
2. **Mutate**: Flip random bits in parallel on the GPU (default: 10% of total bits)
3. **Evaluate**: Calculate fitness (Hamming distance to target "HELLO WORLD")
4. **Select**: Keep changes if fitness improved or stayed same, otherwise revert
5. **Repeat**: Continue until target is found or max iterations reached

## Features

- Parallel bit mutations using CUDA
- Configurable mutation rate and data size
- Simple fitness function (Hamming distance)
- Automatic rollback on regression
- Real-time progress display

## Configurable Parameters

Edit these in [main.cu](main.cu):

```cpp
#define DATA_SIZE 11           // RAM size in bytes
#define MUTATION_RATE 0.1f     // 10% mutation rate
#define MAX_ITERATIONS 10000   // Maximum evolution iterations
```

## Build Instructions

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- CMake 3.18 or higher
- C++ compiler

### Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Run

```bash
./evolving_memory
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
