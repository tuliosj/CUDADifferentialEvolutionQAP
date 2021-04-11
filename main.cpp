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

//
//  testMain.cpp
//
// This is a test code to show an example usage of Differential Evolution

#include <stdio.h>

#include "DifferentialEvolution.hpp"
#include <string.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

instance *open_instance(const char *instance_name) {
    // Find and open the resource files
	FILE *data, *soln;
	char data_path[BUFFER_LENGTH], soln_path[BUFFER_LENGTH];
    sprintf(data_path, "./res/qapdata/%s.dat", instance_name);
    sprintf(soln_path, "./res/qapsoln/%s.sln", instance_name);

    if ((data = fopen(data_path, "r")) == NULL)
    {
        std::cout << "Data file not found." << std::endl;
        exit(1);
    };
    if ((soln = fopen(soln_path, "r")) == NULL)
    {
        std::cout << "Solution file not found." << std::endl;
        exit(1);
    };

    // Instance
    instance *inst;
    cudaMallocManaged(&inst, sizeof(instance));

    // Find instance's size
    char buffer[BUFFER_LENGTH];
    fgets(buffer, BUFFER_LENGTH, data);
    inst->n = atoi(strtok(buffer, "\n"));

    // // Memory allocation
	inst->distance;
    cudaMallocManaged(&inst->distance, sizeof(int)*inst->n*inst->n);
	inst->flow;
    cudaMallocManaged(&inst->flow, sizeof(int)*inst->n*inst->n);
    inst->best_individual;
    cudaMallocManaged(&inst->best_individual, sizeof(int)*inst->n);

    // // Read the distance matrix
	char *token;
	int i,j;
    fgets(buffer, BUFFER_LENGTH, data);

    for (i=0;i<inst->n && fgets(buffer, BUFFER_LENGTH, data);i++) {
        token = strtok(buffer, " ");
        for (j=0;j<inst->n && token != NULL;j++) {
            token[strcspn(token, "\n")] = 0;
            inst->distance[i*inst->n + j] = atoi(token);
            token = strtok(NULL, " ");
        }
    }

    // Read the flow matrix
    fgets(buffer, BUFFER_LENGTH, data);

    for (i=0;i<inst->n && fgets(buffer, BUFFER_LENGTH, data);i++) {
        token = strtok(buffer, " ");
        for (j=0;j<inst->n  && token != NULL;j++) {
            token[strcspn(token, "\n")] = 0;
            inst->flow[i*inst->n + j] = atoi(token);
            token = strtok(NULL, " ");
        }
    }

    // Read the solution file
    fgets(buffer, BUFFER_LENGTH, soln);
    
    strtok(buffer, " ");
    inst->best_result = atoi(strtok(NULL, "\n"));

    fgets(buffer, BUFFER_LENGTH, soln);
    token = strtok(buffer, " ");
    for (i=0;i<inst->n && token != NULL;i++) {
        token[strcspn(token, "\n")] = 0;
        inst->best_individual[i] = atoi(token);
        token = strtok(NULL, " ");
    }

	return inst;
}


int *relativePositionIndexingCPU(const float *vec, int n) {
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
    
    return vecRPI;
}

int printResult(const float *vec, const struct instance *inst) {
    int i, j, sum=0;

    std::cout << "Result (before RPI) = ";
    for(i=0;i<inst->n-1;i++)
        std::cout << vec[i] << ", ";
    std::cout << vec[i] << std::endl;

    int *vecRPI = relativePositionIndexingCPU(vec, inst->n);

    std::cout << "Result (after RPI) = ";
    for(i=0;i<inst->n-1;i++)
        std::cout << vecRPI[i] << ", ";
    std::cout << vecRPI[i] << std::endl;

    for(i=0;i<inst->n;i++) {
        for(j=0;j<inst->n;j++) {
            sum += inst->flow[(vecRPI[i]-1)*inst->n + (vecRPI[j]-1)] * inst->distance[i*inst->n + j];
        }
    }

    std::cout << "Cost = " <<  sum << std::endl;

    return sum;
}

int main(void)
{
    // data that is created in host, then copied to a device version for use with the cost function.
    const struct instance *inst = open_instance("chr12a");



    // create the min and max bounds for the search space.
    int i;
    float minBounds[inst->n];
    float maxBounds[inst->n];
    for(i=0;i<inst->n;i++) {
        minBounds[i] = -3.0;
        maxBounds[i] = 3.0;
    }
    
    // Create the minimizer with a popsize of 192, 100 generations, Dimensions = 2, CR = 0.9, F = 2
    DifferentialEvolution minimizer(92, 1500, inst->n, 0.9, 0.7, minBounds, maxBounds);
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // get the result from the minimizer
    float *result = minimizer.fmin(inst);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printResult(result, inst);

    std::cout << "Finished main function." << std::endl;
    std::cout << "Time difference (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
    return 1;
}
