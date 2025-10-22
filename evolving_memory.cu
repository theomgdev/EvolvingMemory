#include "evolving_memory.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// CUDA kernel to backup data (32-bit optimized)
__global__ void backupData(unsigned char* data, unsigned char* backup, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process 32-bit words for better memory throughput
    unsigned int* data32 = (unsigned int*)data;
    unsigned int* backup32 = (unsigned int*)backup;
    int numWords = dataSize / 4;

    if (idx < numWords) {
        backup32[idx] = data32[idx];
    }

    // Handle remaining bytes
    int remainingStart = numWords * 4;
    int remainingIdx = remainingStart + idx;
    if (remainingIdx < dataSize) {
        backup[remainingIdx] = data[remainingIdx];
    }
}

// CUDA kernel to mutate random bits in parallel (32-bit optimized)
__global__ void mutateBits(unsigned char* data, int dataSize, float mutationRate, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Work with 32-bit words for better performance
    unsigned int* data32 = (unsigned int*)data;
    int numWords = dataSize / 4;  // Number of 32-bit words
    int totalBits = dataSize * 8;
    int bitsToMutate = (int)(totalBits * mutationRate);

    if (idx < bitsToMutate) {
        // Initialize random state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate random bit position
        int bitPos = curand(&state) % totalBits;
        int wordIdx = bitPos / 32;  // Which 32-bit word
        int bitIdx = bitPos % 32;    // Which bit in that word

        // Flip the bit using 32-bit operation (atomic for safety)
        if (wordIdx < numWords) {
            unsigned int mask = 1u << bitIdx;
            atomicXor(&data32[wordIdx], mask);
        } else {
            // Handle remaining bytes (if dataSize not multiple of 4)
            // Use atomicXor on int-aligned address with proper masking
            int byteIdx = bitPos / 8;
            int byteBitIdx = bitPos % 8;
            if (byteIdx < dataSize) {
                // For unaligned bytes, use regular XOR with atomic on nearest word
                int wordBase = byteIdx / 4;
                int byteOffset = byteIdx % 4;
                unsigned int fullMask = (1u << byteBitIdx) << (byteOffset * 8);
                if (wordBase < numWords + 1 && byteIdx < dataSize) {
                    atomicXor(&data32[wordBase], fullMask);
                }
            }
        }
    }
}

// CUDA kernel to restore backup if fitness regressed (32-bit optimized)
__global__ void restoreBackup(unsigned char* data, unsigned char* backup, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process 32-bit words for better memory throughput
    unsigned int* data32 = (unsigned int*)data;
    unsigned int* backup32 = (unsigned int*)backup;
    int numWords = dataSize / 4;

    if (idx < numWords) {
        data32[idx] = backup32[idx];
    }

    // Handle remaining bytes
    int remainingStart = numWords * 4;
    int remainingIdx = remainingStart + idx;
    if (remainingIdx < dataSize) {
        data[remainingIdx] = backup[remainingIdx];
    }
}

// Initialize evolving memory context
EvolvingMemoryContext* evolvingMemoryInit(const EvolvingMemoryConfig& config) {
    EvolvingMemoryContext* ctx = new EvolvingMemoryContext();

    ctx->dataSize = config.dataSize;
    ctx->mutationRate = config.mutationRate;
    ctx->fitnessFunc = config.fitnessFunc;
    ctx->fitnessKernel = config.fitnessKernel;
    ctx->userData = config.userData;
    ctx->d_userData = config.d_userData;

    // Determine which fitness method to use
    if (ctx->fitnessKernel != nullptr) {
        ctx->useGpuFitness = true;
    } else if (ctx->fitnessFunc != nullptr) {
        ctx->useGpuFitness = false;
    } else {
        fprintf(stderr, "Error: Either fitnessFunc or fitnessKernel is required\n");
        delete ctx;
        return nullptr;
    }

    // Allocate device memory for fitness result if using GPU fitness
    if (ctx->useGpuFitness) {
        cudaError_t err = cudaMalloc(&ctx->d_fitnessResult, sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "Error allocating device memory for fitness result: %s\n", cudaGetErrorString(err));
            delete ctx;
            return nullptr;
        }
    } else {
        ctx->d_fitnessResult = nullptr;
    }

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&ctx->d_data, ctx->dataSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for data: %s\n", cudaGetErrorString(err));
        delete ctx;
        return nullptr;
    }

    err = cudaMalloc(&ctx->d_backup, ctx->dataSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for backup: %s\n", cudaGetErrorString(err));
        cudaFree(ctx->d_data);
        delete ctx;
        return nullptr;
    }

    // Initialize with random data
    unsigned char* h_init = new unsigned char[ctx->dataSize];
    srand(time(NULL));
    for (int i = 0; i < ctx->dataSize; i++) {
        h_init[i] = rand() % 256;
    }
    cudaMemcpy(ctx->d_data, h_init, ctx->dataSize, cudaMemcpyHostToDevice);
    delete[] h_init;

    // Setup grid/block dimensions
    int bitsToMutate = (int)(ctx->dataSize * 8 * ctx->mutationRate);
    ctx->mutationBlockSize = dim3(256);
    ctx->mutationGridSize = dim3((bitsToMutate + ctx->mutationBlockSize.x - 1) / ctx->mutationBlockSize.x);

    // For backup/restore, calculate based on max(32-bit words, remaining bytes)
    int numWords = ctx->dataSize / 4;
    int maxElements = (numWords > ctx->dataSize) ? numWords : ctx->dataSize;
    ctx->dataBlockSize = dim3(256);
    ctx->dataGridSize = dim3((maxElements + ctx->dataBlockSize.x - 1) / ctx->dataBlockSize.x);

    // Calculate initial fitness
    if (ctx->useGpuFitness) {
        // Use GPU kernel
        ctx->fitnessKernel(ctx->d_data, ctx->d_fitnessResult, ctx->dataSize, ctx->d_userData);
        cudaDeviceSynchronize();
        cudaMemcpy(&ctx->currentFitness, ctx->d_fitnessResult, sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        // Use CPU function
        unsigned char* h_data = new unsigned char[ctx->dataSize];
        cudaMemcpy(h_data, ctx->d_data, ctx->dataSize, cudaMemcpyDeviceToHost);
        ctx->currentFitness = ctx->fitnessFunc(h_data, ctx->dataSize, ctx->userData);
        delete[] h_data;
    }

    return ctx;
}

// Free evolving memory context
void evolvingMemoryFree(EvolvingMemoryContext* ctx) {
    if (ctx) {
        if (ctx->d_data) cudaFree(ctx->d_data);
        if (ctx->d_backup) cudaFree(ctx->d_backup);
        if (ctx->d_fitnessResult) cudaFree(ctx->d_fitnessResult);
        delete ctx;
    }
}

// Perform one iteration of evolution
float evolvingMemoryEvolveOnce(EvolvingMemoryContext* ctx, int iteration) {
    // Step 1: Backup current data before mutation
    backupData<<<ctx->dataGridSize, ctx->dataBlockSize>>>(ctx->d_data, ctx->d_backup, ctx->dataSize);
    cudaDeviceSynchronize();

    // Step 2: Mutate random bits in parallel
    unsigned long long seed = (unsigned long long)time(NULL) + iteration;
    mutateBits<<<ctx->mutationGridSize, ctx->mutationBlockSize>>>(
        ctx->d_data, ctx->dataSize, ctx->mutationRate, seed);
    cudaDeviceSynchronize();

    // Step 3: Evaluate fitness
    float newFitness;
    if (ctx->useGpuFitness) {
        // Use GPU kernel (no device-to-host copy needed!)
        ctx->fitnessKernel(ctx->d_data, ctx->d_fitnessResult, ctx->dataSize, ctx->d_userData);
        cudaDeviceSynchronize();
        cudaMemcpy(&newFitness, ctx->d_fitnessResult, sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        // Use CPU function (requires device-to-host copy)
        unsigned char* h_data = new unsigned char[ctx->dataSize];
        cudaMemcpy(h_data, ctx->d_data, ctx->dataSize, cudaMemcpyDeviceToHost);
        newFitness = ctx->fitnessFunc(h_data, ctx->dataSize, ctx->userData);
        delete[] h_data;
    }

    // Step 4: If regression, restore backup
    if (newFitness > ctx->currentFitness) {
        // Regression - undo changes
        restoreBackup<<<ctx->dataGridSize, ctx->dataBlockSize>>>(ctx->d_data, ctx->d_backup, ctx->dataSize);
        cudaDeviceSynchronize();
    } else {
        // Progress or same - keep changes
        ctx->currentFitness = newFitness;
    }

    return ctx->currentFitness;
}

    // Run evolution loop until optimal fitness is found or max iterations reached
int evolvingMemoryEvolve(EvolvingMemoryContext* ctx, int maxIterations, bool verbose) {
    if (verbose) {
        printf("Initial state:\n");
        evolvingMemoryPrintState(ctx, 0);
        printf("\nStarting evolution...\n\n");
    }

    for (int iter = 1; iter <= maxIterations; iter++) {
        float previousFitness = ctx->currentFitness;
        float newFitness = evolvingMemoryEvolveOnce(ctx, iter);

        // Print progress when fitness improves
        if (verbose && newFitness < previousFitness) {
            evolvingMemoryPrintState(ctx, iter);
        }

        // Check if we reached optimal fitness
        if (newFitness == 0.0f) {
            if (verbose) {
                printf("\n*** SUCCESS! Achieved optimal fitness in %d iterations ***\n", iter);
            }
            return iter;
        }
    }

    return -1; // Optimal fitness not achieved
}

// Get current data state
void evolvingMemoryGetData(EvolvingMemoryContext* ctx, unsigned char* output) {
    cudaMemcpy(output, ctx->d_data, ctx->dataSize, cudaMemcpyDeviceToHost);
}

// Print current state
void evolvingMemoryPrintState(EvolvingMemoryContext* ctx, int iteration) {
    unsigned char* h_data = new unsigned char[ctx->dataSize];
    cudaMemcpy(h_data, ctx->d_data, ctx->dataSize, cudaMemcpyDeviceToHost);

    printf("Iteration %d: \"", iteration);
    for (int i = 0; i < ctx->dataSize; i++) {
        if (h_data[i] >= 32 && h_data[i] <= 126) {
            printf("%c", h_data[i]);
        } else {
            printf(".");
        }
    }
    printf("\" | Fitness: %.0f\n", ctx->currentFitness);

    delete[] h_data;
}
