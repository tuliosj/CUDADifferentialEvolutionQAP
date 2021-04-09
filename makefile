test:
	nvcc -o output testMain.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu

main:
	nvcc -o output main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu

clean:
	
