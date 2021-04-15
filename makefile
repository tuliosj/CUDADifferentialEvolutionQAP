main:
	g++ -pthread -o output main.cpp DifferentialEvolution.cpp  DifferentialEvolutionGPU.cpp -O3

clean:
	
