#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>

#define NUM_THREADS 	743511 	// length of calculation
#define BLOCK_SIZE 	256	// number of threads per block used in gpu calc
#define EPS		0.00005 // Epsilon for tolerance of diffs between cpu and gpu calculations
#define INCLUDE_MEMTIME false	// Decides whether to include memory transfers to and from gpu in gpu timing
#define PRINTLINES	0	// Number of lines to print in output during validation

__global__ void calcKernel(float* d_in, float *d_out) {
	const unsigned int lid = threadIdx.x;			// local id inside a block
	const unsigned int gid = blockIdx.x*blockDim.x + lid; 	// global id
	d_out[gid] = pow((d_in[gid] / ( d_in[gid] - 2.3 )),3);	// do computation
}


int timeval_subtract(  struct timeval* result, 
                       struct timeval* t2,
                       struct timeval* t1) {
	unsigned int resolution = 1000000;
	long int diff = (t2->tv_usec + resolution * t2->tv_sec) -
                        (t1->tv_usec + resolution * t1->tv_sec);
	result->tv_sec = diff / resolution;
	result->tv_usec = diff % resolution;
	return (diff<0);
}

unsigned int long gpucalc(float* h_in, float* h_out, unsigned int mem_size, unsigned int num_threads) {
	struct timeval t_start, t_end, t_diff;
	struct timeval t_startmem, t_endmem, t_diffmem;

	// device configuration
	unsigned int block_size   = BLOCK_SIZE;
	unsigned int num_blocks   = ((num_threads + (block_size - 1)) / block_size);
	
	// allocate device memory
	float* d_in;
	float* d_out;
	cudaMalloc((void**)&d_in,  mem_size);
	cudaMalloc((void**)&d_out, mem_size);

	gettimeofday(&t_startmem, NULL);
	// copy host memory to device
	cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

	// time and execute the kernel
	gettimeofday(&t_start, NULL);
	calcKernel<<< num_blocks, block_size >>>(d_in, d_out);
	cudaThreadSynchronize();
	gettimeofday(&t_end, NULL);

	// copy result from device to host
	cudaMemcpy(h_out, d_out, sizeof(float)*num_threads, cudaMemcpyDeviceToHost );
	gettimeofday(&t_endmem, NULL);

	// clean up memory
	cudaFree(d_in);	cudaFree(d_out);

	timeval_subtract(&t_diff, &t_end, &t_start);
	timeval_subtract(&t_diffmem, &t_endmem, &t_startmem);

	if (INCLUDE_MEMTIME) {
		return (t_diffmem.tv_sec*1e6+t_diffmem.tv_usec);	// microseconds
	} else {
		return (t_diff.tv_sec*1e6+t_diff.tv_usec); 	// microseconds
	}
}

unsigned long int cpucalc(float* h_in, float* h_out, unsigned int calcsize) {
	struct timeval t_start, t_end, t_diff;

	gettimeofday(&t_start, NULL);	
	for(unsigned int i=0; i<calcsize; i++) {
		h_out[i] = pow((h_in[i] / (h_in[i] - 2.3)),3);
	}
	gettimeofday(&t_end, NULL);

	timeval_subtract(&t_diff, &t_end, &t_start);	
	return  t_diff.tv_sec*1e6+t_diff.tv_usec;	// microseconds
}

int main(int argc, char** argv) {
	unsigned int num_threads = NUM_THREADS;
	unsigned int mem_size    = num_threads*sizeof(float);

	unsigned long int cputime, gputime;

	float        maxdev    = 0;
	unsigned int maxdevidx = 0;

	// allocate host memory
	float* h_in     = (float*) malloc(mem_size);
	float* h_outgpu = (float*) malloc(mem_size);
	float* h_outcpu = (float*) malloc(mem_size);
	
	// initialize the memory
	for(unsigned int i = 0; i < num_threads; i++) {
		h_in[i] = (float)i+1;
	}

	// prepare timing and get the calculations done
	gputime = gpucalc(h_in, h_outgpu, mem_size, num_threads);

	cputime = cpucalc(h_in, h_outcpu, num_threads);


	// print and validate result
	int printevry = 1;
	if (PRINTLINES>0) {
		printevry = (NUM_THREADS / PRINTLINES);		
		printf("cpu\t\tgpu\n");
	}
	for(unsigned int i=0; i<num_threads; i++) {
		if (i % printevry == 1) {printf("%.6f\t%.6f\n", h_outcpu[i], h_outgpu[i]);}
		if (maxdev < abs(h_outcpu[i] - h_outgpu[i])) {
			maxdev = abs(h_outcpu[i]-h_outgpu[i]);
			maxdevidx = i;
		}
	}
	if (maxdev < EPS) {printf("VALID, max deviation: %.6f at calculation no. %d\n", maxdev, maxdevidx);}
	else {printf("INVALID, max deviation: %.6f\t at %d\n", maxdev, maxdevidx);}

	printf("Time for cpu calculation: %d microseconds (%.2f ms)\n",cputime, cputime/1000.0);
	printf("Time for gpu calculation: %d microseconds (%.2f ms)\n",gputime, gputime/1000.0);
	

	// clean up memory
	free(h_in);	free(h_outgpu);	free(h_outcpu);
	
}
