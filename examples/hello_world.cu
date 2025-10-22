#include "../evolving_memory.cuh"
#include <stdio.h>
#include <string.h>

// Target string for the Hello World example
const char TARGET[] = "HELLO WORLD";

/**
 * CPU-based fitness function that calculates Hamming distance
 * Lower score is better, 0.0 means perfect match
 *
 * This is a simple example demonstrating CPU-based fitness evaluation.
 * The library will copy data from device to host for each fitness evaluation.
 */
float calculateHammingDistance(const unsigned char* data, int dataSize, void* userData) {
    const char* target = (const char*)userData;

    int distance = 0;
    for (int i = 0; i < dataSize; i++) {
        unsigned char xorResult = data[i] ^ target[i];
        // Count set bits (Hamming distance)
        while (xorResult) {
            distance += xorResult & 1;
            xorResult >>= 1;
        }
    }

    return (float)distance;
}

int main() {
    printf("=== Evolving Memory - Hello World Example ===\n");
    printf("This example uses CPU-based fitness evaluation\n");
    printf("Target: \"%s\"\n", TARGET);
    printf("Data Size: %d bytes\n", (int)strlen(TARGET));
    printf("Mutation Rate: 10.0%%\n");
    printf("Max Iterations: 10000\n");
    printf("=================================================\n\n");

    // Configure evolving memory with CPU fitness function
    EvolvingMemoryConfig config = {0};
    config.dataSize = strlen(TARGET);
    config.mutationRate = 0.1f;
    config.maxIterations = 10000;
    config.fitnessFunc = calculateHammingDistance;  // CPU function
    config.fitnessKernel = nullptr;                 // No GPU kernel
    config.userData = (void*)TARGET;                // Host memory pointer
    config.d_userData = nullptr;                    // No device memory

    // Initialize evolving memory
    EvolvingMemoryContext* ctx = evolvingMemoryInit(config);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize evolving memory\n");
        return 1;
    }

    printf("Using CPU-based fitness function (requires device-to-host transfers)\n\n");

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

    return 0;
}
