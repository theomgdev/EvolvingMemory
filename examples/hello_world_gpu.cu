#include "../evolving_memory.cuh"
#include <stdio.h>
#include <string.h>

// Target string for the Hello World example
const char TARGET[] = "HELLO WORLD";

/**
 * GPU kernel to calculate Hamming distance using 32-bit chunking
 * Each thread processes one 32-bit word (4 bytes) for better memory throughput
 * Uses __popc() intrinsic for fast population count
 */
__global__ void hammingDistanceKernel(const unsigned char* d_data, const unsigned char* d_target,
                                       int* d_distances, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Cast to 32-bit pointers for word-level processing
    const unsigned int* data32 = (const unsigned int*)d_data;
    const unsigned int* target32 = (const unsigned int*)d_target;
    int numWords = dataSize / 4;

    int distance = 0;

    // Process 32-bit words (4 bytes at a time)
    if (idx < numWords) {
        unsigned int xorResult = data32[idx] ^ target32[idx];
        distance = __popc(xorResult);  // Fast hardware popcount
        d_distances[idx] = distance;
    }
    // Handle remaining bytes (if dataSize not multiple of 4)
    else {
        int remainingStart = numWords * 4;
        int byteIdx = remainingStart + (idx - numWords);

        if (byteIdx < dataSize) {
            unsigned char xorResult = d_data[byteIdx] ^ d_target[byteIdx];
            distance = __popc(xorResult);
            d_distances[idx] = distance;
        } else {
            d_distances[idx] = 0;
        }
    }
}

/**
 * Reduction kernel to sum up partial distances using shared memory
 */
__global__ void reduceSum(int* d_input, float* d_output, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? d_input[i] : 0;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(d_output, (float)sdata[0]);
    }
}

/**
 * Initialize result to zero
 */
__global__ void initResultKernel(float* d_result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_result = 0.0f;
    }
}

/**
 * GPU-based fitness kernel wrapper
 * This is the function passed to the library
 *
 * Demonstrates full GPU-based fitness evaluation with optimizations:
 * 1. Calculates Hamming distance using 32-bit chunking (4 bytes per thread)
 * 2. Uses __popc() hardware intrinsic for fast bit counting
 * 3. Reduces partial results using shared memory (minimal atomic contention)
 * 4. No device-to-host memory transfer needed during evolution!
 *
 * Performance: ~4x faster memory access, scales well to GB-sized data
 */
void calculateHammingDistanceGPU(const unsigned char* d_data, float* d_result,
                                  int dataSize, void* d_userData) {
    const unsigned char* d_target = (const unsigned char*)d_userData;

    // Calculate number of elements to process (32-bit words + remaining bytes)
    int numWords = dataSize / 4;
    int remainingBytes = dataSize % 4;
    int totalElements = numWords + remainingBytes;

    // Allocate temporary device memory for partial distances
    int* d_distances;
    cudaMalloc(&d_distances, totalElements * sizeof(int));

    // Initialize result to 0
    initResultKernel<<<1, 1>>>(d_result);
    cudaDeviceSynchronize();

    // Calculate Hamming distance using 32-bit chunking
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    hammingDistanceKernel<<<gridSize, blockSize>>>(d_data, d_target, d_distances, dataSize);
    cudaDeviceSynchronize();

    // Reduce partial distances to final sum using shared memory
    int sharedMemSize = blockSize * sizeof(int);
    reduceSum<<<gridSize, blockSize, sharedMemSize>>>(d_distances, d_result, totalElements);
    cudaDeviceSynchronize();

    // Free temporary memory
    cudaFree(d_distances);
}

int main() {
    printf("=== Evolving Memory - Hello World GPU Example ===\n");
    printf("This example uses GPU-based fitness evaluation\n");
    printf("Target: \"%s\"\n", TARGET);
    printf("Data Size: %d bytes\n", (int)strlen(TARGET));
    printf("Mutation Rate: 10.0%%\n");
    printf("Max Iterations: 10000\n");
    printf("=================================================\n\n");

    // Allocate and copy target to device memory
    unsigned char* d_target;
    cudaMalloc(&d_target, strlen(TARGET));
    cudaMemcpy(d_target, TARGET, strlen(TARGET), cudaMemcpyHostToDevice);

    // Configure evolving memory with GPU fitness kernel
    EvolvingMemoryConfig config = {0};
    config.dataSize = strlen(TARGET);
    config.mutationRate = 0.1f;
    config.maxIterations = 10000;
    config.fitnessFunc = nullptr;                        // No CPU function
    config.fitnessKernel = calculateHammingDistanceGPU;  // Use GPU kernel
    config.userData = nullptr;                           // No host data
    config.d_userData = d_target;                        // Device memory pointer

    // Initialize evolving memory
    EvolvingMemoryContext* ctx = evolvingMemoryInit(config);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize evolving memory\n");
        cudaFree(d_target);
        return 1;
    }

    printf("Using GPU-based fitness kernel (no device-to-host transfers)\n\n");

    // Run evolution with verbose output
    int iterations = evolvingMemoryEvolve(ctx, config.maxIterations, true);

    // Print final state
    printf("\n");
    if (iterations > 0) {
        printf("*** SUCCESS! Found target in %d iterations ***\n\n", iterations);
        printf("Final state:\n");
        evolvingMemoryPrintState(ctx, iterations);

        // Get and verify the final data
        unsigned char* finalData = new unsigned char[config.dataSize];
        evolvingMemoryGetData(ctx, finalData);
        printf("\nFinal data: \"");
        for (int i = 0; i < config.dataSize; i++) {
            printf("%c", finalData[i]);
        }
        printf("\"\n");
        delete[] finalData;
    } else {
        printf("Failed to find target within %d iterations\n", config.maxIterations);
        printf("Final state:\n");
        evolvingMemoryPrintState(ctx, config.maxIterations);
    }

    // Cleanup
    evolvingMemoryFree(ctx);
    cudaFree(d_target);

    return 0;
}
