#include "evolving_memory.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// CUDA kernel to backup data
__global__ void backupData(unsigned char* data, unsigned char* backup, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        backup[idx] = data[idx];
    }
}

// CUDA kernel to mutate random bits in parallel
__global__ void mutateBits(unsigned char* data, int dataSize, float mutationRate, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalBits = dataSize * 8;
    int bitsToMutate = (int)(totalBits * mutationRate);

    if (idx < bitsToMutate) {
        // Initialize random state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate random bit position
        int bitPos = curand(&state) % totalBits;
        int byteIdx = bitPos / 8;
        int bitIdx = bitPos % 8;

        // Flip the bit - use simple XOR (safe since we mutate different bits each iteration)
        unsigned char mask = 1 << bitIdx;
        data[byteIdx] ^= mask;
    }
}

// CUDA kernel to restore backup if fitness regressed
__global__ void restoreBackup(unsigned char* data, unsigned char* backup, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        data[idx] = backup[idx];
    }
}

// Initialize evolving memory context
EvolvingMemoryContext* evolvingMemoryInit(const EvolvingMemoryConfig& config) {
    EvolvingMemoryContext* ctx = new EvolvingMemoryContext();

    ctx->dataSize = config.dataSize;
    ctx->mutationRate = config.mutationRate;
    ctx->fitnessFunc = config.fitnessFunc;
    ctx->userData = config.userData;

    // Verify fitness function is provided
    if (ctx->fitnessFunc == nullptr) {
        fprintf(stderr, "Error: Fitness function is required\n");
        delete ctx;
        return nullptr;
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
    ctx->dataBlockSize = dim3(256);
    ctx->dataGridSize = dim3((ctx->dataSize + ctx->dataBlockSize.x - 1) / ctx->dataBlockSize.x);

    // Calculate initial fitness using user-provided function
    unsigned char* h_data = new unsigned char[ctx->dataSize];
    cudaMemcpy(h_data, ctx->d_data, ctx->dataSize, cudaMemcpyDeviceToHost);
    ctx->currentFitness = ctx->fitnessFunc(h_data, ctx->dataSize, ctx->userData);
    delete[] h_data;

    return ctx;
}

// Free evolving memory context
void evolvingMemoryFree(EvolvingMemoryContext* ctx) {
    if (ctx) {
        if (ctx->d_data) cudaFree(ctx->d_data);
        if (ctx->d_backup) cudaFree(ctx->d_backup);
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

    // Step 3: Evaluate fitness using user-provided function
    unsigned char* h_data = new unsigned char[ctx->dataSize];
    cudaMemcpy(h_data, ctx->d_data, ctx->dataSize, cudaMemcpyDeviceToHost);
    float newFitness = ctx->fitnessFunc(h_data, ctx->dataSize, ctx->userData);
    delete[] h_data;

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

// Run evolution loop until target is found or max iterations reached
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

        // Check if we found the solution
        if (newFitness == 0.0f) {
            if (verbose) {
                printf("\n*** SUCCESS! Found target in %d iterations ***\n", iter);
            }
            return iter;
        }
    }

    return -1; // Target not found
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
