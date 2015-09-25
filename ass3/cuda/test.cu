#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hostlib.cu.h"

// in general
#define EPS 0.0005
#define BLOCK_SIZE 512
// for warmup
#define NUM_THREADS 7624
// task one specific
#define COLS_N 32
#define ROWS_M 32
// task two specific
#define N2 32
#define M2 64 // As per assignment
// task three specific
#define N 32 
#define M 32
#define U 32


// declaration of functions used in main
// ALL should be moved to hostlib and implemented there
bool validate(float* ground_truth, float* same);    // TODO

int transpose_cpu(int rows_in, int cols_in, float *m_in, float *m_out);       // TODO
int transpose_gpu_naive(int rows_in, int cols_in, float *m_in, float *m_out); // TODO
int transpose_gpu(int rows_in, int cols_in, float *m_in, float *m_out);       // TODO

int matrix_accfun_cpu(int rows_in, int cols_in, float* m_in, float* m_out_a);        // TODO 
int matrix_accfun_gpu_first(int rows_in, int cols_in, float* m_in, float* m_out_a);  // TODO
int matrix_accfun_gpu_second(int rows_in, int cols_in, float* m_in, float* m_out_a); // TODO

int matmult_cpu(int M, int U, float* m_in_a, int U, int N, float* m_in_b, float* m_out_a);     // TODO
int matmult_gpu(int M, int U, float* m_in_a, int U, int N, float* m_in_b, float* m_out_a);     // TODO
int matmult_gpu_opt(int M, int U, float* m_in_a, int U, int N, float* m_in_b, float* m_out_a); // TODO



// declared with purpose of starting the file with its main function
int warmup();
int task_one();
int task_two();
int task_three();


int main(int argc, char** argv) {
    warmup(); // sole purpose is 'warming up GPU' so that timings get valid downstream.
    task_one();
    task_two();
    task_three();
}




int warmup(){
    // performing max segment sum calculation for GPU warmup purpose
    const unsigned int block_size  = BLOCK_SIZE;
    int* h_in    = (int*) malloc( NUM_THREADS * sizeof(int));

    for(unsigned int i=0; i<NUM_THREADS; i++) h_in[i] = 1;

    { // calling maxSegmentSum
        int* d_in;
        cudaMalloc((void**)&d_in , NUM_THREADS * sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size_int, cudaMemcpyHostToDevice);

        // run function on GPU
        maxSegmentSum ( block_size, num_threads, d_in );
        
        // cleanup memory
        cudaFree(d_in );
    }

    // cleanup memory
    free(h_in );

    return 0;
}



int task_one(){
    // Transpose Matrix 
    // 1a. implement serial version
    // 1b. bonus objective implement in OPENMP
    // 1c. implement naive  version
    // 1d. implement serial version

    // initiate data to transpose (dense matrix)
    float *h_in, *h_out_a, *h_out_b, *h_out_c, *h_out_d;
    m_in = malloc(COLS_N * ROWS_M * sizeof(float));
    m_out_a = malloc(COLS_N * ROWS_M * sizeof(float));
    m_out_c = malloc(COLS_N * ROWS_M * sizeof(float));
    m_out_d = malloc(COLS_N * ROWS_M * sizeof(float));

    for (int i=0; i<(COLS_N*ROWS_M); i++){
        m_in[i] = 0 // TODO random number
    }

    // initiate timing variable, keep results for validation
    unsigned long int elapsed_a, elapsed_b, elapsed_c, elapsed_d;
    struct timeval t_start, t_end, t_diff;
    bool valid_c, valid_d; 
    
    // TASK 1 A)
    { 
        gettimeofday(&t_start, NULL); 

        transpose_cpu(ROWS_M, COLS_N, m_in, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_a = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    }
    printf("Transpose Matrix sized %d x %d on CPU runs in: %lu microsecs", COLS_N, ROWS_M, elapsed_a);
    
    // TASK 1 B) OMITTED

    // TASK 1 C)
    { 
        gettimeofday(&t_start, NULL); 

        transpose_gpu_naive(ROWS_M, COLS_N, m_in, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_c = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_c = validate(h_out_a, h_out_c);
    }
    printf("Transpose Matrix sized %d x %d on GPU naïvely runs in: %lu microsecs\n", COLS_N, ROWS_M, elapsed_c);
    if (valid_c) printf("Naïve implementation is VALID\n");
    else printf("Naïve implementation is INVALID\n");


    // TASK 1 D)
    { 
        gettimeofday(&t_start, NULL); 

        transpose_gpu(ROWS_M, COLS_N, m_in, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_d = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_d = validate(h_out_a, h_out_d);
    }
    printf("Transpose Matrix sized %d x %d on GPU optimized runs in: %lu microsecs", COLS_N, ROWS_M, elapsed_d);
    if (valid_d) printf("Optimal implementation is VALID\n");
    else printf("Optimal implementation is INVALID\n");

    // TODO print statistics, speedup difference and so on

    free(h_in);
    free(h_out_a);
    free(h_out_c);
    free(h_out_d);

    return 0;
}




int task_two(){
    // Matrix Transposition during or as a pre-computatiion
    // 2 a. Reason about loop-level parallellism 
    // 2 b. bonus objective implement in OPENMP
    // 2 c. implement QUICKLY straightforward cuda
    // 2 d. Rewrite QUICKLY to coalesced global mem access

    // initiate data to transpose (dense matrix)
    const int m = 64;
    float *h_in, *h_out_a, *h_out_b, *h_out_c, *h_out_d;
    m_in = malloc(M2 * N2 * sizeof(float));
    m_out_a = malloc(M2 * N2 * sizeof(float));
    m_out_c = malloc(M2 * N2 * sizeof(float));
    m_out_d = malloc(M2 * N2 * sizeof(float));

    for (int i=0; i<(M2*N2); i++){
        m_in[i] = 0 // TODO random number
    }

    // initiate timing variable, keep results for validation
    unsigned long int elapsed_a, elapsed_b, elapsed_c, elapsed_d;
    struct timeval t_start, t_end, t_diff;
    bool valid_c, valid_d; 
    
    // TASK 1 A)
    { 
        gettimeofday(&t_start, NULL); 

        matrix_accfun_cpu(M2, N2, m_in, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_a = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    }
    printf("Matrix accfun on size %d x %d on CPU runs in: %lu microsecs",M2, N2, elapsed_a);
    
    // TASK 1 B) OMITTED

    // TASK 1 C)
    { 
        gettimeofday(&t_start, NULL); 

        matrix_accfun_gpu_first(M2, N2, m_in, m_out_c); 
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_c = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_c = validate(h_out_a, h_out_c);
    }
    printf("Matrix accfun on size %d x %d on GPU first impl runs in: %lu microsecs\n",M2, N2, elapsed_c);
    if (valid_c) printf("Implementation is VALID\n");
    else printf("Implementation is INVALID\n");


    // TASK 1 D)
    { 
        gettimeofday(&t_start, NULL); 

        matrix_accfun_gpu_first(M2, N2, m_in, m_out_d);  
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_d = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_d = validate(h_out_a, h_out_d);
    }
    printf("Matrix accfun on size %d x %d on GPU rewrite runs in: %lu microsecs",M2, N2, elapsed_d);
    if (valid_d) printf("Implementation is VALID\n");
    else printf("Implementation is INVALID\n");

    // TODO print statistics, speedup difference and so on

    free(h_in);
    free(h_out_a);
    free(h_out_c);
    free(h_out_d);

    return 0;
}



int task_three(){
    // Dense Matrix-Matrix multiplication
    // 1a. implement serial version
    // 1b. bonus objective implement in OPENMP
    // 1c. implement naive  version
    // 1d. implement serial version

    // initiate data to transpose (dense matrix)
    float *m_in_a, *m_in_b, *m_out_a, *m_out_c, *m_out_d;
    m_in_a = malloc(M * U * sizeof(float));
    m_in_b = malloc(U * N * sizeof(float));
    m_out_a = malloc(M * N * sizeof(float));
    m_out_c = malloc(M * N * sizeof(float));
    m_out_d = malloc(M * N * sizeof(float));
    
    for (int i=0; i<(COLS_N*ROWS_M); i++){
        m_in[i] = 0; // TODO random number
    }

    // initiate timing variable, keep results for validation
    unsigned long int elapsed_a, elapsed_b, elapsed_c, elapsed_d;
    struct timeval t_start, t_end, t_diff;
    bool valid_c, valid_d; 
    
    // TASK 1 A)
    { 
        gettimeofday(&t_start, NULL); 

        matmult_cpu(M, U, m_in_a, U, N, m_in_b, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_a = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    }
    printf("MatrixMult on (%dx%d) x (%d,%d) on CPU runs in: %lu microsecs",M,U,U,N, elapsed_a);
    
    // TASK 1 B) OMITTED

    // TASK 1 C)
    { 
        gettimeofday(&t_start, NULL); 

        matmult_gpu(M, U, m_in_a, U, N, m_in_b, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_c = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_c = validate(h_out_a, h_out_c);
    }
    printf("MatrixMult on (%dx%d) x (%d,%d) on GPU runs in: %lu microsecs",M,U,U,N, elapsed_c);
    if (valid_c) printf("Implementation is VALID\n");
    else printf("Implementation is INVALID\n");


    // TASK 1 D)
    { 
        gettimeofday(&t_start, NULL); 

        matmult_gpu_opt(M, U, m_in_a, U, N, m_in_b, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_d = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_d = validate(h_out_a, h_out_d);
    }
    printf("MatrixMult on (%dx%d) x (%d,%d) on CPU runs in: %lu microsecs",M,U,U,N, elapsed_d);
    if (valid_d) printf("Optimal Implementation is VALID\n");
    else printf("Optimal Implementation is INVALID\n");

    // TODO print statistics, speedup difference and so on

    free(h_in);
    free(h_out_a);
    free(h_out_c);
    free(h_out_d);

    return 0;
}

