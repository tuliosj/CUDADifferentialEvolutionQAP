#include <thread>
#include <random>
#include <iostream>
#include <ctime> // for clock()
#include <cmath>
#include <algorithm>
#include <iterator>
#include <atomic>

struct instance
{
    int n;
	int *distance;
	int *flow;
	int *best_individual;
	int best_result;
};

int *relativePositionIndexing(const float *vec, int n) {
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

int costFunction(const float *vec, const struct instance *inst, std::atomic<long int> *costCalls) {
    int i, j, sum=0;

    *costCalls++; // costCalls refers to how many times the cost function was calculated

    int *vecRPI = relativePositionIndexing(vec, inst->n);

    for(i=0;i<inst->n;i++) { // Cost function
        for(j=0;j<inst->n;j++) {
            sum += inst->flow[(vecRPI[i]-1)*inst->n + (vecRPI[j]-1)] * inst->distance[i*inst->n + j];
        }
    }

    free(vecRPI);

    return sum;
}

void swap(float *vec, int i, int j) {
    float aux = vec[i];
    vec[i] = vec[j];
    vec[j] = aux;
}

void swapReinsert(float *vec, int i, int j) {
    float aux = vec[i];
    int k;

    for(k=i;k<j;k++) { // Indexes inbetween go back a index
        vec[k] = vec[k+1];
    }
    vec[k] = aux; // And vec[i] goes to the index j
}

void swapReinsertReverse(float *vec, int i, int j) {
    float aux = vec[j];
    int k;

    for(k=j;k>i;k--) {
        vec[k] = vec[k-1];
    }
    vec[k] = aux;
}

void swap2opt(float *vec, int i, int j) {
    int c=(j-i+1)/2; // 2-opt swaps the indexes between i and j, so the number of swaps is half the distance

    for(int k=0;k<c;k++)
        swap(vec,i++,j--);
}

int localSearch(float *vec, const struct instance *inst, std::atomic<long int> *costCalls) {
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

void generatePopulation(float *d_x, float d_min, float d_max,
            int *costs, const struct instance *inst, 
            int popSize, int idx, std::atomic<long int> *costCalls)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    for (int i = 0; i < inst->n; i++) {
        std::uniform_real_distribution<> dist(0.0,1.0);
        d_x[(idx*inst->n) + i] = (dist(rng) * (d_max - d_min)) + d_min;
    }

    costs[idx] = costFunction(&d_x[idx*inst->n], inst, costCalls);
}

void evolutionKernel(float *d_pop, float *d_trial, int *costs, float *d_nextPop,
    int popSize, int CR, float F, const struct instance *inst, int idx, std::atomic<long int> *costCalls) {
    
    std::random_device dev;
    std::mt19937 rng(dev());
    
    int a, b, c, j;

    std::uniform_int_distribution<std::mt19937::result_type> distPop(0,popSize-1);
    std::uniform_int_distribution<std::mt19937::result_type> distDim(0,inst->n-1);
    std::uniform_int_distribution<std::mt19937::result_type> distThousand(0,999);

    // Indexes for mutation
    do { a = distPop(rng); } while (a == idx);
    do { b = distPop(rng); } while (b == idx || b == a);
    do { c = distPop(rng); } while (c == idx || c == a || c == b);
    j = distDim(rng);
    
    for (int i = 0; i <= inst->n; i++) {
        if (distThousand(rng) < CR || j==i) { // If crossover is satisfied, it mutates the current index
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
    const struct instance *inst, std::atomic<long int> *costCalls, long int maxCostCalls) {

    int CR = crossoverRate*1000;
    // Allocation of values
    float *d_pop = (float *)malloc(sizeof(float)*popSize*inst->n);
    float *d_nextPop = (float *)malloc(sizeof(float)*popSize*inst->n);
    float *d_trial = (float *)malloc(sizeof(float)*popSize*inst->n);
    int *costs = (int *)malloc(sizeof(int)*popSize*inst->n);

    // "First of all, your thread block size should always be a multiple of 32, 
    // because kernels issue instructions in warps (32 threads). 
    // For example, if you have a block size of 50 threads, the GPU will still
    // issue commands to 64 threads and you'd just be wasting them."
    // https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
    int popSize32 = ceil(popSize / 32.0) * 32;

    // Generate the population
    std::thread threadList[popSize];
    int i;
    for(i=0;i<popSize;i++) {
        threadList[i] = std::thread(generatePopulation, d_pop, d_min, d_max, costs, inst, popSize, i, costCalls);
    }
    for(i=0;i<popSize;i++) {
        threadList[i].join();
    }
    
    for (int i = 1; i <= maxGenerations && *costCalls <= maxCostCalls; i++) {

        // start kernel for this generation
        for(i=0;i<popSize;i++) {
            threadList[i] = std::thread(evolutionKernel, d_pop, d_trial, costs, d_nextPop, popSize, CR, F, inst, i, costCalls);
        }
        for(i=0;i<popSize;i++) {
            threadList[i].join();
        }

        // Update the population for the next generation
        float *tmp = d_pop;
        d_pop = d_nextPop;
        d_nextPop = tmp;
    }

    int best_index = std::distance(costs, std::min_element(costs, costs+popSize));
    return d_pop+(best_index*inst->n);
}
