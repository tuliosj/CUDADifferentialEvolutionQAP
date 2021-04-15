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

#include <stdio.h>

#include "DifferentialEvolution.hpp"
#include <string.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <charconv>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>


struct instance *open_instance(const std::string instance_name) {
    // Instance
    struct instance *inst;
    cudaMallocManaged(&inst, sizeof(instance));

    std::string splitStr;
    int number;
    unsigned long long int count = -1;
    std::ifstream data("./res/qapdata/" + instance_name + ".dat");
    if (data.is_open()) {
        while(std::getline(data,splitStr,' ')) {
            if (splitStr[0]!='\0') { 
                std::from_chars(splitStr.data(), splitStr.data()+splitStr.size(), number);
                if(count==-1) {
                    inst->n = number;
                    inst->distance;
                    cudaMallocManaged(&inst->distance, sizeof(int)*inst->n*inst->n);
                    inst->flow;
                    cudaMallocManaged(&inst->flow, sizeof(int)*inst->n*inst->n);
                    inst->best_individual;
                    cudaMallocManaged(&inst->best_individual, sizeof(int)*inst->n);
                } else if(count < inst->n*inst->n) {
                    inst->distance[count] = number;
                } else if(count < 2*inst->n*inst->n) {
                    inst->flow[count - inst->n*inst->n] = number;
                } else {
                    std::cout << "File read" << std::endl;
                    break;
                }
                count++;
            }

        }
        data.close();
    } else {
        std::cout << "Unable to open file";
        exit(1);
    }

    std::ifstream sln("./res/qapsoln/" + instance_name + ".sln");
    if (sln.is_open()) {
        count = -2;
        while(std::getline(sln,splitStr,' ')) {
            if (splitStr[0]!='\0') { 
                std::from_chars(splitStr.data(), splitStr.data()+splitStr.size(), number);
                if(count==-2) {
                } else if(count == -1) {
                    inst->best_result= number;
                } else if(count < inst->n) {
                    inst->best_individual[count] = number;
                } else {
                    std::cout << "File read" << std::endl;
                    break;
                }
                count++;
            }

        }
        sln.close();
    } else {
        std::cout << "Unable to open file";
        exit(1);
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


int writeDown(const float *vec, const struct instance *inst, std::string instName, std::ofstream& results, double elapsed, unsigned long long int *costCalls, int popSize, int generations, float f, float cr, std::time_t executingSince) {
    int i, j, sum=0;


    int *vecRPI = relativePositionIndexingCPU(vec, inst->n);

    for(i=0;i<inst->n;i++) {
        for(j=0;j<inst->n;j++) {
            sum += inst->flow[(vecRPI[i]-1)*inst->n + (vecRPI[j]-1)] * inst->distance[i*inst->n + j];
        }
    }
    
    std::cout << "Writing on file -  Inst: " << instName << "\t Gap: " << 100.0*(sum-inst->best_result)/inst->best_result << "\t Time: " << elapsed << std::endl;
    std::cout << "[";
    for(i=0;i<inst->n-1;i++)
        std::cout << vec[i] << ", ";
    std::cout << vec[i] << "]" << std::endl;

    // instance
    results << instName << "\t";
    // gap
    results << 100.0*(sum-inst->best_result)/inst->best_result << "\t";
    // costCalls
    results << costCalls[0] << "\t";
    // time
    results << elapsed << "\t";
    // popSize
    results << popSize << "\t";
    // generations
    results << generations << "\t";
    // f
    results << f << "\t";
    // cr
    results << cr << "\t";

    // result
    results << "[";
    for(i=0;i<inst->n-1;i++)
        results << vecRPI[i] << ", ";
    results << vecRPI[i] << "]\t";
    // best
    results << sum << "\t";

    // datetime
    std::stringstream ss;
    ss << std::put_time(localtime(&executingSince), "%F %H:%M:%S");
    results << ss.str() << "\t";
    // device
    results << "gpu" << "\t";

    results << "\n";
    
    return sum;
}

int main(void)
{
    // std::string str;
    // std::cin >> str;
    // // data that is created in host, then copied to a device version for use with the cost function.
    // const struct instance *inst = open_instance(str);

    // Create .tsv file
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(localtime(&now), "[%F]-[%H:%M:%S]");
    std::ofstream results("./res/results/gpu" + ss.str() + ".tsv");
    results << "instance\tgap\tcostcall\ttime\tpopsize\tgenerations\tf\tcr\tresult\tbest\tdatetime\tdevice\n";

    // Test parameters
    std::string str[] = {"els19"};
    int i, popSize = 320, generations = 100;
    float cr = 0.9, f = 0.7;

    unsigned long long int *costCalls;
    cudaMallocManaged(&costCalls, sizeof(unsigned long long int));
    
    for(int j=0;j<10;j++) {
        for(int k=0;k<1;k++) {
            const struct instance *inst = open_instance(str[k]);

            // Create the minimizer with bounds of -3.0 and 3.0
            DifferentialEvolution minimizer(popSize, generations, inst->n, cr, f, -3.0, 3.0);
            
            *costCalls = 0;
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            // get the result from the minimizer
            float *result = minimizer.fmin(inst, costCalls);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            writeDown(result, inst, str[k], results, std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.0, costCalls, popSize, generations, f, cr, now);
        }
    }

    results.close();

    return 1;
}
