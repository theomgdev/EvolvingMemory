#ifndef EVOLVING_MEMORY_CUH
#define EVOLVING_MEMORY_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Forward declaration
struct EvolvingMemoryContext;

/**
 * @brief Fitness function callback type
 * @param data Current data buffer
 * @param dataSize Size of data in bytes
 * @param userData User-provided data pointer (can be NULL)
 * @return Fitness score (lower is better, 0.0 is perfect)
 */
typedef float (*FitnessFunction)(const unsigned char* data, int dataSize, void* userData);

/**
 * @brief Configuration structure for evolving memory
 */
struct EvolvingMemoryConfig {
    int dataSize;                   // Size of data in bytes
    float mutationRate;             // Mutation rate (0.0 to 1.0)
    int maxIterations;              // Maximum evolution iterations
    FitnessFunction fitnessFunc;    // User-provided fitness function
    void* userData;                 // User data passed to fitness function (optional)
};

/**
 * @brief Context structure for evolving memory operations
 */
struct EvolvingMemoryContext {
    unsigned char* d_data;      // Device memory for data
    unsigned char* d_backup;    // Device memory for backup
    int dataSize;               // Size of data in bytes
    float mutationRate;         // Mutation rate
    FitnessFunction fitnessFunc;// Fitness function callback
    void* userData;             // User data for fitness function
    float currentFitness;       // Current fitness score
    dim3 mutationBlockSize;     // Block size for mutation kernel
    dim3 mutationGridSize;      // Grid size for mutation kernel
    dim3 dataBlockSize;         // Block size for data operations
    dim3 dataGridSize;          // Grid size for data operations
};

// CUDA kernels
/**
 * @brief Backup data kernel
 */
__global__ void backupData(unsigned char* data, unsigned char* backup, int dataSize);

/**
 * @brief Mutate random bits in parallel
 */
__global__ void mutateBits(unsigned char* data, int dataSize, float mutationRate, unsigned long long seed);

/**
 * @brief Restore backup if fitness regressed
 */
__global__ void restoreBackup(unsigned char* data, unsigned char* backup, int dataSize);

// API functions
/**
 * @brief Initialize evolving memory context
 * @param config Configuration parameters
 * @return Pointer to initialized context, or NULL on error
 */
EvolvingMemoryContext* evolvingMemoryInit(const EvolvingMemoryConfig& config);

/**
 * @brief Free evolving memory context
 * @param ctx Context to free
 */
void evolvingMemoryFree(EvolvingMemoryContext* ctx);

/**
 * @brief Perform one iteration of evolution
 * @param ctx Evolving memory context
 * @param iteration Current iteration number
 * @return New fitness score
 */
float evolvingMemoryEvolveOnce(EvolvingMemoryContext* ctx, int iteration);

/**
 * @brief Run evolution loop until target is found or max iterations reached
 * @param ctx Evolving memory context
 * @param maxIterations Maximum number of iterations
 * @param verbose Print progress if true
 * @return Number of iterations taken, or -1 if target not found
 */
int evolvingMemoryEvolve(EvolvingMemoryContext* ctx, int maxIterations, bool verbose);

/**
 * @brief Get current data state
 * @param ctx Evolving memory context
 * @param output Output buffer (must be at least dataSize bytes)
 */
void evolvingMemoryGetData(EvolvingMemoryContext* ctx, unsigned char* output);

/**
 * @brief Print current state
 * @param ctx Evolving memory context
 * @param iteration Current iteration number
 */
void evolvingMemoryPrintState(EvolvingMemoryContext* ctx, int iteration);

#endif // EVOLVING_MEMORY_CUH
