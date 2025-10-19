#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Configurable parameters
#define DATA_SIZE 11           // RAM size in bytes (globally adjustable)
#define MUTATION_RATE 0.1f     // 10% mutation rate (globally adjustable)
#define MAX_ITERATIONS 10000   // Maximum evolution iterations

// Target string for fitness evaluation
const char TARGET[] = "HELLO WORLD";

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

// Fitness function: calculate Hamming distance to target
// Lower distance = better fitness
__host__ float calculateFitness(unsigned char* d_data, int dataSize) {
    unsigned char h_data[DATA_SIZE];
    cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost);

    int distance = 0;
    for (int i = 0; i < dataSize; i++) {
        unsigned char xorResult = h_data[i] ^ TARGET[i];
        // Count set bits (Hamming distance)
        while (xorResult) {
            distance += xorResult & 1;
            xorResult >>= 1;
        }
    }

    return (float)distance;
}

// Print current state
void printState(unsigned char* d_data, int dataSize, int iteration, float fitness) {
    unsigned char h_data[DATA_SIZE];
    cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost);

    printf("Iteration %d: \"", iteration);
    for (int i = 0; i < dataSize; i++) {
        if (h_data[i] >= 32 && h_data[i] <= 126) {
            printf("%c", h_data[i]);
        } else {
            printf(".");
        }
    }
    printf("\" | Fitness: %.0f\n", fitness);
}

int main() {
    printf("=== Evolving Memory - Simple MVP ===\n");
    printf("Target: \"%s\"\n", TARGET);
    printf("Data Size: %d bytes\n", DATA_SIZE);
    printf("Mutation Rate: %.1f%%\n", MUTATION_RATE * 100);
    printf("=====================================\n\n");

    // Allocate device memory
    unsigned char *d_data, *d_backup;
    cudaMalloc(&d_data, DATA_SIZE);
    cudaMalloc(&d_backup, DATA_SIZE);

    // Initialize with random data
    unsigned char h_init[DATA_SIZE] = {0};
    for (int i = 0; i < DATA_SIZE; i++) {
        h_init[i] = rand() % 256;
    }
    cudaMemcpy(d_data, h_init, DATA_SIZE, cudaMemcpyHostToDevice);

    // Calculate initial fitness
    float previousFitness = calculateFitness(d_data, DATA_SIZE);
    printf("Initial state:\n");
    printState(d_data, DATA_SIZE, 0, previousFitness);
    printf("\nStarting evolution...\n\n");

    // Evolution loop
    int bitsToMutate = (int)(DATA_SIZE * 8 * MUTATION_RATE);
    dim3 mutationBlockSize(256);
    dim3 mutationGridSize((bitsToMutate + mutationBlockSize.x - 1) / mutationBlockSize.x);
    dim3 dataBlockSize(256);
    dim3 dataGridSize((DATA_SIZE + dataBlockSize.x - 1) / dataBlockSize.x);

    for (int iter = 1; iter <= MAX_ITERATIONS; iter++) {
        // Step 1: Backup current data before mutation
        backupData<<<dataGridSize, dataBlockSize>>>(d_data, d_backup, DATA_SIZE);
        cudaDeviceSynchronize();

        // Step 2: Mutate random bits in parallel
        unsigned long long seed = (unsigned long long)time(NULL) + iter;
        mutateBits<<<mutationGridSize, mutationBlockSize>>>(d_data, DATA_SIZE, MUTATION_RATE, seed);
        cudaDeviceSynchronize();

        // Step 3: Evaluate fitness
        float currentFitness = calculateFitness(d_data, DATA_SIZE);

        // Step 4: If regression, restore backup
        if (currentFitness > previousFitness) {
            // Regression - undo changes
            restoreBackup<<<1, DATA_SIZE>>>(d_data, d_backup, DATA_SIZE);
            cudaDeviceSynchronize();
        } else {
            // Progress or same - keep changes
            if (currentFitness < previousFitness) {
                // Print when we make progress
                printState(d_data, DATA_SIZE, iter, currentFitness);
            }
            previousFitness = currentFitness;

            // Check if we found the solution
            if (currentFitness == 0.0f) {
                printf("\n*** SUCCESS! Found target in %d iterations ***\n", iter);
                break;
            }
        }
    }

    printf("\nFinal state:\n");
    printState(d_data, DATA_SIZE, MAX_ITERATIONS, previousFitness);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_backup);

    return 0;
}
