test:
	g++ -pthread -o output main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cpp

main:
	g++ -pthread -o output main.cpp DifferentialEvolution.cpp  DifferentialEvolutionGPU.cpp

clean:
	
