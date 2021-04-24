#include <stdio.h>
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

struct instance
{
    int n;
	int *distance;
	int *flow;
	int *best_individual;
	int best_result;
};

float *differentialEvolution(float d_min, float d_max, int popSize, int maxGenerations, int crossoverRate,  float F, 
    const struct instance *inst, unsigned long long int *costCalls, long int maxCostCalls);

struct instance *open_instance(const std::string instance_name) {
    
    // Instance allocation
    struct instance *inst;
    cudaMallocManaged(&inst, sizeof(instance));

    // Variables for handling the data stream
    std::string splitStr;
    int number;
    long int current_index = -1;
    std::ifstream data("./res/qapdata/" + instance_name + ".dat");

    // Open file
    if (data.is_open()) {
        // While there's some thing to read
        while(std::getline(data,splitStr,' ')) {
            if (splitStr[0]!='\0') { 
                std::from_chars(splitStr.data(), splitStr.data()+splitStr.size(), number);
                if(current_index==-1) { // First number is n; Arrays are allocated accordingly.
                    inst->n = number;
                    inst->distance;
                    cudaMallocManaged(&inst->distance, sizeof(int)*inst->n*inst->n);
                    inst->flow;
                    cudaMallocManaged(&inst->flow, sizeof(int)*inst->n*inst->n);
                    inst->best_individual;
                    cudaMallocManaged(&inst->best_individual, sizeof(int)*inst->n);
                } else if(current_index < inst->n*inst->n) { // Read distance matrix n^2
                    inst->distance[current_index] = number;
                } else if(current_index < 2*inst->n*inst->n) { // Read flow matrix n^2
                    inst->flow[current_index - inst->n*inst->n] = number;
                } else {
                    std::cout << "Nothing else to read -- Is this file correct?" << std::endl;
                    break;
                }
                current_index++;
            }
        }
        data.close();
    } else {
        std::cout << "Unable to open data file for " << instance_name << std::endl;
        exit(1);
    }

    // Same for solution file
    std::ifstream sln("./res/qapsoln/" + instance_name + ".sln");
    if (sln.is_open()) {
        current_index = -2;
        while(std::getline(sln,splitStr,' ')) {
            if (splitStr[0]!='\0') { 
                std::from_chars(splitStr.data(), splitStr.data()+splitStr.size(), number);
                if(current_index==-2) { // First number is n, but we already have it
                } else if(current_index == -1) { // Best result
                    inst->best_result = number;
                } else if(current_index < inst->n) { // Best solution
                    inst->best_individual[current_index] = number;
                } else {
                    std::cout << "Nothing else to read -- Is this file correct?" << std::endl;
                    break;
                }
                current_index++;
            }
        }
        sln.close();
    } else {
        std::cout << "Unable to open solution file for " << instance_name << std::endl;
        exit(1);
    }
    
    return inst;
}

void printInstance(const struct instance *inst) {
    int i,j;
    std::cout << "n: " << inst->n << std::endl;

    std::cout << "distance: {";
    for(i=0; i<inst->n; i++) {
        std::cout << "[";
        for(j=0; j<inst->n-1; j++) {
            std::cout << inst->distance[i*inst->n + j] << ", ";            
        }
        std::cout << inst->distance[i*inst->n + j] << "]" <<  std::endl;
    }
    std::cout << "}" << std::endl;

    std::cout << "flow: {";
    for(i=0; i<inst->n; i++) {
        std::cout << "[";
        for(j=0; j<inst->n-1; j++) {
            std::cout << inst->flow[i*inst->n + j] << ", ";            
        }
        std::cout << inst->flow[i*inst->n + j] << "]" <<  std::endl;
    }
    std::cout << "}" << std::endl;

    std::cout << "best: " << inst->best_result << std::endl;
    
    std::cout << "solution: [";
    for(j=0; j<inst->n-1; j++) {
        std::cout << inst->best_individual[j] << ", ";            
    }
    std::cout << inst->best_individual[j] << "]" <<  std::endl;
}


int *relativePositionIndexingMain(const float *vec, int n) { // This is exactly like the __device__ function in the .cu file
    
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


int writeDown(const float *vec, const struct instance *inst, std::string instName, std::string file, double elapsed, unsigned long long int *costCalls, int popSize, int generations, float f, float cr, std::time_t executingSince) {
    
    int i, j, sum=0;

    // RPI
    int *vecRPI = relativePositionIndexingMain(vec, inst->n);

    for(i=0;i<inst->n;i++) { // Cost function
        for(j=0;j<inst->n;j++) {
            sum += inst->flow[(vecRPI[i]-1)*inst->n + (vecRPI[j]-1)] * inst->distance[i*inst->n + j];
        }
    }
    
    std::cout << "Writing on file -  Inst: " << instName << "\t Gap: " << 100.0*(sum-inst->best_result)/inst->best_result << "\t Time: " << elapsed << std::endl;
    std::cout << "[";
    for(i=0;i<inst->n-1;i++)
        std::cout << vec[i] << ", ";
    std::cout << vec[i] << "]" << std::endl;

    std::ofstream results(file, std::ios::app);

    // instance
    results << inst->n << "\t";
    // gap
    results << 100.0*(sum-inst->best_result)/inst->best_result << "\t";
    // costCalls
    results << *costCalls << "\t";
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
    results << "gpu";

    results << "\n";

    results.close();
    
    return sum;
}

int main(void) {

    // Create .tsv file
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(localtime(&now), "[%F]-[%H:%M:%S]");
    std::string file = "./res/results/gpu-" + ss.str() + ".tsv";
    std::ofstream results(file);
    results << "instance\tgap\tcostcalls\ttime\tpopsize\tgenerations\tf\tcr\tresult\tbest\tdatetime\tdevice\n"; // Header
    results.close();

    // Experiment parameters
    std::string instName[] = {"chr12c","els19","bur26a","ste36a","tho40","sko56","lipa60a"};
    //,"tai64c","sko72","lipa90a","wil100","esc128","tho150","tai256c"};
    int popSize = 1024, generations = 100, executions = 10;
    float cr = 0.9, f = 0.7;

    unsigned long long int *costCalls;
    cudaMallocManaged(&costCalls, sizeof(unsigned long long int));
    
    for(int j=0;j<executions;j++) {
        for(int k=0;k<sizeof(instName)/sizeof(std::string);k++) {
            const struct instance *inst = open_instance(instName[k]);

            // printInstance(inst);
            
            *costCalls = 0; // costCalls refers to how many times the cost function was calculated

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            // Execute the DE based on within the initial bounds of -3.0 and 3.0; The last parameter is how many costCalls until it stops.
            float *result = differentialEvolution(-3.0, 3.0, popSize, generations, cr, f, inst, costCalls, 150000000);

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            writeDown(result, inst, instName[k], file, std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.0, costCalls, popSize, generations, f, cr, now);
        }
    }

    return 1;
}
