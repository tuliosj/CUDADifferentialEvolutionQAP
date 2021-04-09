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
    instance *inst = (instance*)malloc(sizeof(instance));

    // Find instance's size
    char buffer[BUFFER_LENGTH];
    fgets(buffer, BUFFER_LENGTH, data);
    inst->n = atoi(strtok(buffer, "\n"));

    // // Memory allocation
	inst->distance = (int*)malloc(sizeof(int)*inst->n*inst->n);
	inst->flow = (int*)malloc(sizeof(int)*inst->n*inst->n);
    inst->best_individual  = (int*)malloc(sizeof(int)*inst->n);

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

int main(void)
{
    std::cout << open_instance("chr12a")->best_result << std::endl;

    // create the min and max bounds for the search space.
    float minBounds[2] = {-50, -50};
    float maxBounds[2] = {100, 200};
    
    // a random array or data that gets passed to the cost function.
    float arr[3] = {2.5, 2.6, 2.7};
    
    // data that is created in host, then copied to a device version for use with the cost function.
    struct data x;
    struct data *d_x;
    gpuErrorCheck(cudaMalloc(&x.arr, sizeof(float) * 3));
    unsigned long size = sizeof(struct data);
    gpuErrorCheck(cudaMalloc((void **)&d_x, size));
    x.v = 3;
    x.dim = 2;
    gpuErrorCheck(cudaMemcpy(x.arr, (void *)&arr, sizeof(float) * 3, cudaMemcpyHostToDevice));
    
    // Create the minimizer with a popsize of 192, 50 generations, Dimensions = 2, CR = 0.9, F = 2
    DifferentialEvolution minimizer(192,50, 2, 0.9, 0.5, minBounds, maxBounds);
    
    gpuErrorCheck(cudaMemcpy(d_x, (void *)&x, sizeof(struct data), cudaMemcpyHostToDevice));
    
    // get the result from the minimizer
    std::vector<float> result = minimizer.fmin(d_x);
    std::cout << "Result = " << result[0] << ", " << result[1] << std::endl;
    std::cout << "Finished main function." << std::endl;
    return 1;
}
