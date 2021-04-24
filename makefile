gpu:
	nvcc -o output ./GPU/main.cpp ./GPU/DifferentialEvolution.cu -Xptxas -O3,-v

gpu-p:
	nvcc -o output ./GPU-P/main.cpp ./GPU-P/DifferentialEvolution.cu -Xptxas -O3,-v

cpu:
	g++ -pthread -o output ./CPU/main.cpp ./CPU/DifferentialEvolution.cpp -O3
