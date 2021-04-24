// A basic macro used to checking cuda errors.
// @param ans - the most recent enumerated cuda error to check.
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cfloat> // for FLT_MAX
#include <iostream>
#include <ctime> // for clock()
#include <cmath>
#include <algorithm>
#include <iterator>

struct instance
{
    int n;
	int *distance;
	int *flow;
	int *best_individual;
	int best_result;
};

// Basic function for exiting code on CUDA errors.
// Does no special error handling, just exits the program if it finds any errors and gives an error message.
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


__device__ int *relativePositionIndexing(const float *vec, int n) {
    int i, j;

    int *vecRPI = (int*)malloc(sizeof(int)*n);
    
    for(i=0;i<n;i++) {
        vecRPI[i] = n;
        for(j=0;j<n;j++) {
            // vecRPI[i] gets n minus the amount of indexes greater than vec[i]
            if(i!=j&&vec[j]>vec[i]) {
                vecRPI[i]--;
            }
        }
    }

    // Remove dupes
    for(i=0;i<n;i++) for(j=0;j<n;j++) if(i!=j&&vecRPI[j]==vecRPI[i]) vecRPI[j]--;
    
    return vecRPI;
}

__device__ void relativePositionIndexingP(const float *vec, int n, int *vecRPI) {
    int i, j;

    for(i=0;i<n;i++) {
        vecRPI[i] = n;
        for(j=0;j<n;j++) {
            // vecRPI[i] gets n minus the amount of indexes greater than vec[i]
            if(i!=j&&vec[j]>vec[i]) {
                vecRPI[i]--;
            }
        }
    }

    // Remove dupes
    for(i=0;i<n;i++) for(j=0;j<n;j++) if(i!=j&&vecRPI[j]==vecRPI[i]) vecRPI[j]--;
}

__global__ void costFunctionP(const int *vecRPI, const struct instance *inst, unsigned long long int *costCalls, int *sum, int popSize) {
    int idx = threadIdx.x;
    if (idx >= popSize) return;

    int i = blockIdx.x/inst->n;
    int j = blockIdx.x%inst->n;

    if(i==0 && j==0)
    atomicAdd(costCalls, 1); // costCalls here should only be added once per execution

    atomicAdd(&sum[idx], inst->flow[(vecRPI[idx*inst->n + i]-1)*inst->n + (vecRPI[idx*inst->n + j]-1)] * inst->distance[i*inst->n + j]);
}

__device__ int costFunction(const float *vec, const struct instance *inst, unsigned long long int *costCalls) {
    int i, j, sum=0;

    atomicAdd(costCalls, 1); // costCalls refers to how many times the cost function was calculated

    int *vecRPI = relativePositionIndexing(vec, inst->n);

    for(i=0;i<inst->n;i++) { // Cost function
        for(j=0;j<inst->n;j++) {
            sum += inst->flow[(vecRPI[i]-1)*inst->n + (vecRPI[j]-1)] * inst->distance[i*inst->n + j];
        }
    }

    free(vecRPI);

    return sum;
}

__device__ void swap(float *vec, int i, int j) {
    float aux = vec[i];
    vec[i] = vec[j];
    vec[j] = aux;
}

__device__ void swapReinsert(float *vec, int i, int j) {
    float aux = vec[i];
    int k;

    for(k=i;k<j;k++) { // Indexes inbetween go back a index
        vec[k] = vec[k+1];
    }
    vec[k] = aux; // And vec[i] goes to the index j
}

__device__ void swapReinsertReverse(float *vec, int i, int j) {
    float aux = vec[j];
    int k;

    for(k=j;k>i;k--) {
        vec[k] = vec[k-1];
    }
    vec[k] = aux;
}

__device__ void swap2opt(float *vec, int i, int j) {
    int c=(j-i+1)/2; // 2-opt swaps the indexes between i and j, so the number of swaps is half the distance

    for(int k=0;k<c;k++)
        swap(vec,i++,j--);
}

__device__ int localSearch(float *vec, const struct instance *inst, unsigned long long int *costCalls) {
    int i, j, cost_2;

    int cost = costFunction(vec, inst, costCalls);

    for(i=0;i<inst->n-1;i++) {
        for(j=i+1;j<inst->n;j++) {
            swap2opt(vec,i,j);
            cost_2 = costFunction(vec, inst, costCalls);
            if(cost_2 < cost) {
                return cost_2;
            }
            swap2opt(vec,i,j);
        }
    }

    return cost;
}

void printCudaVector(const float *d_vec, int size)
{
    std::cout << "\nsize: " << size << std::endl;
    float *h_vec = new float[size];
    gpuErrorCheck(cudaMemcpy(h_vec, d_vec, sizeof(float) * size, cudaMemcpyDeviceToHost));
    
    int i;
    std::cout << "{";
    for (i = 0; i < size-1; i++) {
        std::cout << h_vec[i] << ", ";
    }
    std::cout << h_vec[i] << "}" << std::endl;
    
    delete[] h_vec;
}

__global__ void generatePopulation(float *d_x, float d_min, float d_max,
            int *costs, const struct instance *inst, curandState_t *randStates,
            int popSize, unsigned long seed,  unsigned long long int *costCalls)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= popSize) return;
    
    curandState_t *state = &randStates[idx];
    curand_init(seed, idx, 0, state);
    for (int i = 0; i < inst->n; i++) {
        d_x[(idx*inst->n) + i] = (curand_uniform(state) * (d_max - d_min)) + d_min;
    }

    costs[idx] = costFunction(&d_x[idx*inst->n], inst, costCalls);
}

__global__ void evolutionKernelP(float *d_pop, float *d_trial, int *costs, float *d_nextPop, curandState_t *randStates,
    int popSize, int CR, float F, const struct instance *inst, int *vecrpi, int *sum)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= popSize) return; 
    curandState_t *state = &randStates[idx];
    sum[idx] = 0;

    int a, b, c, j;
    // Indexes for mutation
    do { a = curand(state) % popSize; } while (a == idx);
    do { b = curand(state) % popSize; } while (b == idx || b == a);
    do { c = curand(state) % popSize; } while (c == idx || c == a || c == b);
    j = curand(state) % inst->n;

    
    for (int i = 1; i <= inst->n; i++) {
        if ((curand(state) % 1000) < CR || j==i) { // If crossover is satisfied, it mutates the current index
            d_trial[(idx*inst->n)+i] = d_pop[(a*inst->n)+i] + (F * (d_pop[(b*inst->n)+i] - d_pop[(c*inst->n)+i]));
        } else { // If there's no crossover for this index
            d_trial[(idx*inst->n)+i] = d_pop[(idx*inst->n)+i];
        }
    }

    relativePositionIndexingP(&d_trial[idx*inst->n], inst->n, &vecrpi[idx*inst->n]);
}


__global__ void selectionP(float *d_pop, float *d_trial, int *costs, float *d_nextPop, int dim, int popSize, int *score) {

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= popSize) return;

    int j;
    if (score[idx] < costs[idx]) { // Update the next generation with the trial vector
        for (j = 0; j < dim; j++) {
            d_nextPop[(idx*dim) + j] = d_trial[(idx*dim) + j];
        }
        costs[idx] = score[idx];
    } else { // Keep the individual for the next generation
        for (j = 0; j < dim; j++) {
            d_nextPop[(idx*dim) + j] = d_pop[(idx*dim) + j];
        }
    }
}

__global__ void evolutionKernel(float *d_pop, float *d_trial, int *costs, float *d_nextPop, curandState_t *randStates, 
    int popSize, int CR, float F, const struct instance *inst, unsigned long long int *costCalls) {
    
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= popSize) return; 
    curandState_t *state = &randStates[idx];
    
    int a, b, c, j;
    // Indexes for mutation
    do { a = curand(state) % popSize; } while (a == idx);
    do { b = curand(state) % popSize; } while (b == idx || b == a);
    do { c = curand(state) % popSize; } while (c == idx || c == a || c == b);
    j = curand(state) % inst->n;
    
    for (int i = 0; i <= inst->n; i++) {
        if ((curand(state) % 1000) < CR || j==i) { // If crossover is satisfied, it mutates the current index
            d_trial[(idx*inst->n)+i] = d_pop[(a*inst->n)+i] + (F * (d_pop[(b*inst->n)+i] - d_pop[(c*inst->n)+i]));
        } else { // If there's no crossover for this index
            d_trial[(idx*inst->n)+i] = d_pop[(idx*inst->n)+i];
        }
    }
    
    int score = costFunction(&d_trial[idx*inst->n], inst, costCalls);

    if (score < costs[idx]) { // Update the next generation with the trial vector
        for (j = 0; j < inst->n; j++) {
            d_nextPop[(idx*inst->n) + j] = d_trial[(idx*inst->n) + j];
        }
        costs[idx] = score;
    } else { // Keep the individual for the next generation
        for (j = 0; j < inst->n; j++) {
            d_nextPop[(idx*inst->n) + j] = d_pop[(idx*inst->n) + j];
        }
    }
}


float *differentialEvolution(float d_min, float d_max, int popSize, int maxGenerations, int crossoverRate,  float F, 
    const struct instance *inst, unsigned long long int *costCalls, long int maxCostCalls) {

    int CR = crossoverRate*1000;
    // Allocation of values
    float *d_pop, *d_nextPop, *d_trial;
    void *randStates;
    int *costs;
    cudaMallocManaged(&d_pop, sizeof(float) * popSize*inst->n);
    cudaMallocManaged(&d_nextPop, sizeof(float) * popSize*inst->n);
    cudaMallocManaged(&d_trial, sizeof(float) * popSize*inst->n);
    cudaMallocManaged(&randStates, sizeof(curandState_t)*popSize);
    cudaMallocManaged(&costs, sizeof(int) * popSize);

    // "First of all, your thread block size should always be a multiple of 32, 
    // because kernels issue instructions in warps (32 threads). 
    // For example, if you have a block size of 50 threads, the GPU will still
    // issue commands to 64 threads and you'd just be wasting them."
    // https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
    int popSize32 = ceil(popSize / 32.0) * 32;


    // Generate the population
    cudaError_t ret;
    generatePopulation<<<1, popSize32>>>(d_pop, d_min, d_max, costs, inst, (curandState_t *)randStates, popSize, clock(), costCalls);
    gpuErrorCheck(cudaPeekAtLastError());
    ret = cudaDeviceSynchronize();
    gpuErrorCheck(ret);

    int *vecRPI;
    ret = cudaMallocManaged(&vecRPI, sizeof(int) * popSize * inst->n);
    gpuErrorCheck(ret);

    int *sum;
    ret = cudaMallocManaged(&sum, sizeof(int) * popSize);
    gpuErrorCheck(ret);
    
    for (int i = 1; i <= maxGenerations && *costCalls <= maxCostCalls; i++) {

        // start kernel for this generation
        evolutionKernelP<<<1, popSize32>>>(d_pop, d_trial, costs, d_nextPop,
                (curandState_t *)randStates, popSize, CR, F, inst, vecRPI, sum);
        gpuErrorCheck(cudaPeekAtLastError());
        ret = cudaDeviceSynchronize();
        gpuErrorCheck(ret);

        costFunctionP<<<inst->n*inst->n, popSize32>>>(vecRPI, inst, costCalls, sum, popSize);
        gpuErrorCheck(cudaPeekAtLastError());
        ret = cudaDeviceSynchronize();
        gpuErrorCheck(ret);

        selectionP<<<1, popSize32>>>(d_pop, d_trial, costs, d_nextPop, inst->n, popSize, sum);
        gpuErrorCheck(cudaPeekAtLastError());
        ret = cudaDeviceSynchronize();
        gpuErrorCheck(ret);

        // Update the population for the next generation
        float *tmp = d_pop;
        d_pop = d_nextPop;
        d_nextPop = tmp;
    }

    int best_index = std::distance(costs, std::min_element(costs, costs+popSize));
    return d_pop+(best_index*inst->n);
}
