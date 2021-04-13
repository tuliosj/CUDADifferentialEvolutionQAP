test:
	nvcc -o output main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu

main:
	nvcc -o output main.cpp DifferentialEvolution.cpp  DifferentialEvolutionGPU.cu -Xptxas -O3,-v 

clean:
	
