/* Copyright 2017 Ian Rankin
*
* Permission is hereby granted, free of charge, to any person obtaining a copy of this
* software and associated documentation files (the "Software"), to deal in the Software
* without restriction, including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
* to whom the Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or
* substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// DifferentialEvolutionGPU.cu
// This file holds the GPU kernel functions required to run differential evolution.
// The software in this files is based on the paper:
// Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continous Spaces,
// Rainer Storn, Kenneth Price (1996)
//
// But is extended upon for use with GPU's for faster computation times.
// This has been done previously in the paper:
// Differential evolution algorithm on the GPU with C-CUDA
// Lucas de P. Veronese, Renato A. Krohling (2010)
// However this implementation is only vaguly based on their implementation.
// Translation: I saw that the paper existed, and figured that they probably
// implemented the code in a similar way to how I was going to implement it.
// Brief read-through seemed to be the same way.
//
// The paralization in this software is done by using multiple cuda threads for each
// agent in the algorithm. If using smaller population sizes, (4 - 31) this will probably
// not give significant if any performance gains. However large population sizes are more
// likly to give performance gains.
//
// HOW TO USE:
// To implement a new cost function write the cost function in DifferentialEvolutionGPU.cu with the header
// __device float fooCost(const float *vec, const void *inst)
// @param vec - sample parameters for the cost function to give a score on.
// @param inst - any set of arguements that can be passed at the minimization stage
// NOTE: inst any memory given to the function must already be in device memory.
//
// Go to the header and add a specifier for your cost functiona and change the COST_SELECTOR
// to that specifier. (please increment from previous number)
//
// Once you have a cost function find the costFunc function, and add into
// preprocessor directives switch statement
//
// ...
// #elif COST_SELECTOR == YOUR_COST_FUNCTION_SPECIFIER
//      return yourCostFunctionName(vec, inst);
// ...
//


#include <curand_kernel.h>


#include <cuda_runtime.h>
// for random numbers in a kernel
#include "DifferentialEvolutionGPU.h"

// for FLT_MAX
#include <cfloat>

#include <iostream>

// for clock()
#include <ctime>
#include <cmath>

// basic function for exiting code on CUDA errors.
// Does no special error handling, just exits the program if it finds any errors and gives an error message.
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


// -----------------IMPORTANT----------------
// costFunc - this function must implement whatever cost function
// is being minimized.
// Feel free to delete all code in here.
// This is a bit of a hack and not elegant at all. The issue is that
// CUDA doesn't support function passing device code between host
// software. There is a possibilty of using virtual functions, but
// was concerned that the polymorphic function have a lot of overhead
// So instead use this super ugly method for changing the cost function.
//
// @param vec - the vector to be evaulated.
// @param inst - a set of user arguments.

__device__ int *relativePositionIndexing(const float *vec, int n) {
    int i, j, biggerThanMe;

    int *vecRPI = (int*)malloc(sizeof(int)*n);

    float *aux = (float*)malloc(sizeof(float)*n);
    for(i=0;i<n;i++) {
        aux[i] = vec[i];
    }
    
    for(i=0;i<n;i++) {
        biggerThanMe = n;
        for(j=0;j<n;j++) {
            if(i!=j&&aux[j]>aux[i]) {
                biggerThanMe--;
            }
        }
        vecRPI[i]=biggerThanMe;
    }

    // Remove dupes
    for(i=0;i<n;i++) for(j=0;j<n;j++) if(i!=j&&vecRPI[j]==vecRPI[i]) vecRPI[j]--;
    
    free(aux);
    
    return vecRPI;
}

__device__ int costFunc(const float *vec, const struct instance *inst, unsigned long long int *costCalls) {
    int i, j, sum=0;

    atomicAdd(costCalls, 1);

    int *vecRPI = relativePositionIndexing(vec, inst->n);

    for(i=0;i<inst->n;i++) {
        for(j=0;j<inst->n;j++) {
            sum += inst->flow[(vecRPI[i]-1)*inst->n + (vecRPI[j]-1)] * inst->distance[i*inst->n + j];
        }
    }

    free(vecRPI);

    return sum;
}

__device__ void copy(float *a, float *b, int n) {
    for(int i=0;i<n;i++) b[i]=a[i];
}

__device__ void swap(float *vec, int i, int j) {
    float aux = vec[i];
    vec[i] = vec[j];
    vec[j] = aux;
}

__device__ void swapInsertAfter(float *vec, int i, int j) {
    int aux = vec[i];
    for(int k=i+1;k<=j;k++) {
        vec[k-1] = vec[k];
        if(k==j)
            vec[k] = aux;
    }
}

__device__ void swapInsertAfterReverse(float *vec, int i, int j) {
    int aux = vec[j];

    for(int k=j-1;k>=i;k--) {
        vec[k+1] = vec[k];
        if(k==i)
            vec[k] = aux;
    }
}

__device__ void swap2opt(float *vec, int i, int j) {
    int a=i,b=j,c=(j-i+1)/2;
    for(int k=0;k<c;k++)
        swap(vec,a++,b--);
}
// https://www.intechopen.com/books/novel-trends-in-the-traveling-salesman-problem/cuda-accelerated-2-opt-local-search-for-the-traveling-salesman-problem
// __device__ int bestLocalSearch(float *vec, const struct instance *inst, unsigned long long int *costCalls) {
//     int i, j, cost_2, improvementFound = 0;

//     int cost = costFunc(vec, inst, costCalls);

//     int *costs = (int*)malloc(sizeof(int)*inst->n);
//     for(i=0;i<inst->n;i++) {
//         costs[i] = cost;
//     }
    
//     for(i=0;i<inst->n-1;i++) {
//         improvementFound = 0;
//         for(j=i+1;j<inst->n;j++) {
//             swap(vec,i,j);
//             cost_2 = costFunc(vec, inst, costCalls);
//             if(cost_2 < cost) {
//                 costs[j] = cost_2;
//                 improvementFound = 1;
//             }
//             swap(vec,i,j);
//         }
//         if(improvementFound==1) {
//             cost_2 = cost;
//             for(j=i+1;j<inst->n;j++) {
//                 if(costs[j] < cost_2) {
//                     cost_2 = costs[j];
//                     swap(vec,i,j);
//                 }
//             }
//         }
//     }

//     free(costs);

//     return costFunc(vec, inst, costCalls);
// }

__device__ int localSearch(float *vec, const struct instance *inst, unsigned long long int *costCalls) {
    int i, j, cost_2;

    int cost = costFunc(vec, inst, costCalls);

    for(i=0;i<inst->n-1;i++) {
        for(j=i+1;j<inst->n;j++) {
            swap(vec,i,j);
            cost_2 = costFunc(vec, inst, costCalls);
            if(cost_2 < cost) {
                return cost_2;
            }
            swap(vec,i,j);
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

__global__ void generateRandomVectorAndInit(float *d_x, float d_min, float d_max,
            int *d_cost, const struct instance *inst, curandState_t *randStates,
            int popSize, int dim, unsigned long seed,  unsigned long long int *costCalls)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= popSize) return;
    // if (idx >= 1) return;
    
    curandState_t *state = &randStates[idx];
    curand_init(seed, idx,0,state);
    for (int i = 0; i < dim; i++) {
        d_x[(idx*dim) + i] = (curand_uniform(state) * (d_max - d_min)) + d_min;
    }

    d_cost[idx] = costFunc(&d_x[idx*dim], inst, costCalls);
}


// This function handles the entire differentialEvolution, and calls the needed kernel functions.
// @param d_target - a device array with the current agents parameters (requires array with size popSize*dim)
// @param d_trial - a device array with size popSize*dim (worthless outside of function)
// @param d_cost - a device array with the costs of the last generation afterwards size: popSize
// @param d_target2 - a device array with size popSize*dim (worthless outside of function)
// @param d_min - a list of the minimum values for the set of parameters (size = dim)
// @param d_max - a list of the maximum values for the set of parameters (size = dim)
// @param randStates - an array of random number generator states. Array created using createRandNumGen funtion
// @param dim - the number of dimensions the equation being minimized has.
// @param popSize - this the population size for DE, or otherwise the number of agents that DE will use. (see DE paper for more info)
// @param CR - Crossover Constant used by DE (see DE paper for more info)
// @param F - the scaling factor used by DE (see DE paper for more info)
// @param inst - this a set of any arguments needed to be passed to the cost function. (must be in device memory already)
__global__ void evolutionKernel(float *d_target,
                                float *d_trial,
                                int *d_cost,
                                float *d_target2,
                                float d_min,
                                float d_max,
                                curandState_t *randStates,
                                int dim,
                                int popSize,
                                int CR, // Must be given as value between [0,999]
                                float F,
                                const struct instance *inst,
                                unsigned long long int *costCalls)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= popSize) return; // stop executing this block if
                                // all populations have been used
    curandState_t *state = &randStates[idx];
    
    // TODO: Better way of generating unique random numbers?
    int a;
    int b;
    int c;
    int j;
    //////////////////// Random index mutation generation //////////////////
    // select a different random number then index
    do { a = curand(state) % popSize; } while (a == idx);
    do { b = curand(state) % popSize; } while (b == idx || b == a);
    do { c = curand(state) % popSize; } while (c == idx || c == a || c == b);
    j = curand(state) % dim;
    
    ///////////////////// MUTATION ////////////////
    for (int k = 1; k <= dim; k++) {
        if ((curand(state) % 1000) < CR || k==dim) {
            // trial vector param comes from vector plus weighted differential
            d_trial[(idx*dim)+j] = d_target[(a*dim)+j] + (F * (d_target[(b*dim)+j] - d_target[(c*dim)+j]));
        } else {
            d_trial[(idx*dim)+j] = d_target[(idx*dim)+j];
        } // end if else for creating trial vector
        j = (j+1) % dim;
    } // end for loop through parameters
    
    int score = costFunc(&d_trial[idx*dim], inst, costCalls);

    if (score < d_cost[idx]) {
        // copy trial into new vector
        for (j = 0; j < dim; j++) {
            d_target2[(idx*dim) + j] = d_trial[(idx*dim) + j];
            //printf("idx = %d, d_target2[%d] = %f, score = %f\n", idx, (idx*dim)+j, d_trial[(idx*dim) + j], score);
        }
        d_cost[idx] = score;
    } else {
        // copy target to the second vector
        for (j = 0; j < dim; j++) {
            d_target2[(idx*dim) + j] = d_target[(idx*dim) + j];
            //printf("idx = %d, d_target2[%d] = %f, score = %f\n", idx, (idx*dim)+j, d_trial[(idx*dim) + j], score);
        }
    }
} // end differentialEvolution function.


// This is the HOST function that handles the entire Differential Evolution process.
// This function handles the entire differentialEvolution, and calls the needed kernel functions.
// @param d_target - a device array with the current agents parameters (requires array with size popSize*dim)
// @param d_trial - a device array with size popSize*dim (worthless outside of function)
// @param d_cost - a device array with the costs of the last generation afterwards size: popSize
// @param d_target2 - a device array with size popSize*dim (worthless outside of function)
// @param d_min - a list of the minimum values for the set of parameters (size = dim)
// @param d_max - a list of the maximum values for the set of parameters (size = dim)
// @param h_cost - this function once the function is completed will contain the costs of final generation.
// @param randStates - an array of random number generator states. Array created using createRandNumGen funtion
// @param dim - the number of dimensions the equation being minimized has.
// @param popSize - this the population size for DE, or otherwise the number of agents that DE will use. (see DE paper for more info)
// @param maxGenerations - the max number of generations DE will perform (see DE paper for more info)
// @param CR - Crossover Constant used by DE (see DE paper for more info)
// @param F - the scaling factor used by DE (see DE paper for more info)
// @param inst - this a set of any arguments needed to be passed to the cost function. (must be in device memory already)
// @param h_output - the host output vector of function
void differentialEvolution(float *d_target,
                           float *d_trial,
                           int *d_cost,
                           float *d_target2,
                           float d_min,
                           float d_max,
                           int *h_cost,
                           void *randStates,
                           int dim,
                           int popSize,
                           int maxGenerations,
                           int CR, // Must be given as value between [0,999]
                           float F,
                           const struct instance *inst,
                           float *h_output,
                           unsigned long long int *costCalls)
{
    cudaError_t ret;
    // "First of all, your thread block size should always be a multiple of 32, 
    // because kernels issue instructions in warps (32 threads). 
    // For example, if you have a block size of 50 threads, the GPU will still
    // issue commands to 64 threads and you'd just be wasting them."
    // https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
    int power32 = ceil(popSize / 32.0) * 32;
    // std::cout << "power32 = " << power32 << std::endl;
    
    // int i, j;

    // for(i=0;i<inst->n;i++) {
    //     for(j=0;j<inst->n;j++) {
    //         printf("%d ", inst->distance[i*inst->n+j]);
    //     }
    //     printf("\n");
    // }

    // std::cout << "min bounds = ";
    // printCudaVector(d_min, dim);
    // std::cout << "max bounds = ";
    // printCudaVector(d_max, dim);
    
    //std::cout << "Random vector" << std::endl;
    //printCudaVector(d_target, popSize*dim);
    //std::cout << "About to create random vecto" << std::endl;
    
    // generate random vector
    generateRandomVectorAndInit<<<1, power32>>>(d_target, d_min, d_max, d_cost,
                    inst, (curandState_t *)randStates, popSize, dim, clock(), costCalls);
    gpuErrorCheck(cudaPeekAtLastError());
    //udaMemcpy(d_target2, d_target, sizeof(float) * dim * popSize, cudaMemcpyDeviceToDevice);
    
    // ret = cudaDeviceSynchronize();
    // gpuErrorCheck(ret);
    // exit(0);
    // std::cout << "Generated random vector" << std::endl;
    
    // printCudaVector(d_target, popSize*dim);
    // std::cout << "printing cost vector" << std::endl;
    // printCudaVector(d_cost, popSize);
    
    for (int i = 1; i <= maxGenerations; i++) {
        //std::cout << i << ": generation = \n";
        //printCudaVector(d_target, popSize * dim);
        //std::cout << "cost = ";
        //printCudaVector(d_cost, popSize);
        //std::cout << std::endl;
        
        // start kernel for this generation
        evolutionKernel<<<1, power32>>>(d_target, d_trial, d_cost, d_target2, d_min, d_max,
                (curandState_t *)randStates, dim, popSize, CR, F, inst, costCalls);
        gpuErrorCheck(cudaPeekAtLastError());

        
        // swap buffers, places newest data into d_target.
        float *tmp = d_target;
        d_target = d_target2;
        d_target2 = tmp;
    } // end for (generations)
    
    ret = cudaDeviceSynchronize();
    gpuErrorCheck(ret);
    ret = cudaMemcpy(h_cost, d_cost, popSize * sizeof(float), cudaMemcpyDeviceToHost);
    gpuErrorCheck(ret);
    //std::cout << "h_cost = {";
    
    // find min of last evolutions
    int bestIdx = -1;
    float bestCost = FLT_MAX;
    for (int i = 0; i < popSize; i++) {
        float curCost = h_cost[i];
        //std::cout << curCost << ", ";
        if (curCost <= bestCost) {
            bestCost = curCost;
            bestIdx = i;
        }
    }
    //std::cout << "}" << std::endl;
    
    //std::cout << "\n\n agents = ";
    //printCudaVector(d_target, popSize*dim);
    
    //std::cout << "Best cost = " << bestCost << " bestIdx = " << bestIdx << std::endl;
    
    // output best minimization.
    ret = cudaMemcpy(h_output, d_target+(bestIdx*dim), sizeof(float)*dim, cudaMemcpyDeviceToHost);
    gpuErrorCheck(ret);
}

// allocate the memory needed for random number generators.
void *createRandNumGen(int size)
{
    void *x;
    gpuErrorCheck(cudaMalloc(&x, sizeof(curandState_t)*size));
    return x;
}









