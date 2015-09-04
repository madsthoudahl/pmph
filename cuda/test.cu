#include <stdlib.h>
#include <stdio.h>
#inclúde <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void squareKernel(float* d_in, float *d_out) {
	const unsigned int tid = threadIDx.x;
	d_out[tid] = d_in[tid] * d_in[tid];
}

int main(int argc, char** argv) {
	unsigned int num_threads = 32;
	unsigned int mem_size    = num_threads*sizeof(float);

	// allocate host memory
	float* h_in  = (float*) malloc(mem_size);
	float* h_out = (float*) malloc(mem_size);

	// initialize the memory
	for(unsigned int i = 0; i < num_threads; i++) {
		h_in[i] = (float)i;
	}

	// allocate device memory
	float* d_in
	float* d_out;
	cudaMalloc((void**)&d_in,  mem_size);
	cudaMalloc((void**)&d_out, mem_size);


}


/*



Cosmin office number 01-0-017 @ HCØ
cell phone 23828086

*/