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
#define ROWS_M 32
#define COLS_N 32
// task two specific
#define M2 64 // As per assignment
#define N2 32
// task three specific
#define M 32
#define U 32
#define N 32 


//All these functions live in hostlib.cu.h library
//bool validate(int size, float* ground_truth, float* same);    

//void transpose_cpu(int rows_in, int cols_in, float *m_in, float *m_out);
//void transpose_gpu(int rows_in, int cols_in, float *m_in, float *m_out, bool naive);

//void matrix_accfun_cpu(int rows_in, int cols_in, float* m_in, float* m_out_a);
//void matrix_accfun_gpu(int rows_in, int cols_in, float* m_in, float* m_out_a, bool second);

//void matmult_cpu(int M, int U, float* m_in_a, int U, int N, float* m_in_b, float* m_out_a);
//void matmult_gpu(int M, int U, float* m_in_a, int U, int N, float* m_in_b, float* m_out_a, bool opt);



// declared with purpose of starting the file with its main function
int warmup();
void mprinter();
void task_one();
void task_two();
void task_three();


int main(int argc, char** argv) {
    warmup(); // sole purpose is 'warming up GPU' so that timings get valid downstream.
    mprinter();
    //task_one();
    //task_two();
    //task_three();
    return 0;
}




int warmup(){
    // performing max segment sum calculation for GPU warmup purpose
    const unsigned int size        = NUM_THREADS;
    int* h_in    = (int*) malloc( size * sizeof(int));
    int result;

    for(unsigned int i=0; i<size; i++) h_in[i] = 1;

    // run function on GPU
    result = maxSegmentSum_gpu( size, h_in );
    
    return result;
}


void mprinter(){
    const unsigned int rows = 3;
    const unsigned int cols = 3;
    const unsigned int size = rows * cols;
    int* arr = (int*) malloc( size * sizeof(int) );
    for(unsigned int i=0; i<size; i++) h_in[i] = i;
    matprint(rows,cols,arr);
}

void task_one(){
    // Transpose Matrix 
    // 1a. implement serial version
    // 1b. bonus objective implement in OPENMP
    // 1c. implement naive  version
    // 1d. implement serial version

    // initiate data to transpose (dense matrix)
    const unsigned int rows = ROWS_M;
    const unsigned int cols = COLS_N;
    const unsigned int arr_size = rows * cols;

    float *m_in, *m_out_a, *m_out_c, *m_out_d;
    m_in    = (float*) malloc(arr_size * sizeof(float));
    m_out_a = (float*) malloc(arr_size * sizeof(float));
    m_out_c = (float*) malloc(arr_size * sizeof(float));
    m_out_d = (float*) malloc(arr_size * sizeof(float));

    for (int i=0; i<(arr_size); i++){
        m_in[i] = i; // TODO random number
    }

    // initiate timing variable, keep results for validation
    unsigned long int elapsed_a, elapsed_c, elapsed_d;
    struct timeval t_start, t_end, t_diff;
    bool valid_c, valid_d; 
    
    // TASK 1 A)
    { 
        gettimeofday(&t_start, NULL); 

        transpose_cpu<float>( rows, cols, m_in, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_a = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    }
    printf("Transpose Matrix sized %d x %d on CPU runs in: %lu microsecs", cols, rows, elapsed_a);
    
    // TASK 1 B) OMITTED

    // TASK 1 C)
    { 
        gettimeofday(&t_start, NULL); 

        transpose_gpu<float>( rows, cols, m_in, m_out_c, true);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_c = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_c = validate<float>(arr_size, m_out_a, m_out_c);
    }
    printf("Transpose Matrix sized %d x %d on GPU naïvely runs in: %lu microsecs\n", cols, rows, elapsed_c);
    if (valid_c) printf("Naïve implementation is VALID\n");
    else printf("Naïve implementation is INVALID\n");


    // TASK 1 D)
    { 
        gettimeofday(&t_start, NULL); 

        transpose_gpu<float>( rows, cols, m_in, m_out_d, false);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_d = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_d = validate<float>(arr_size, m_out_a, m_out_d);
    }
    printf("Transpose Matrix sized %d x %d on GPU optimized runs in: %lu microsecs", cols, rows, elapsed_d);
    if (valid_d) printf("Optimal implementation is VALID\n");
    else printf("Optimal implementation is INVALID\n");

    // TODO print statistics, speedup difference and so on

    free(m_in);
    free(m_out_a);
    free(m_out_c);
    free(m_out_d);

    return;
}




void task_two(){
    // Matrix Transposition during or as a pre-computatiion
    // 2 a. Reason about loop-level parallellism 
    // 2 b. bonus objective implement in OPENMP
    // 2 c. implement QUICKLY straightforward cuda
    // 2 d. Rewrite QUICKLY to coalesced global mem access

    // initiate data to transpose (dense matrix)
    const int rows = M2;
    const int cols = N2;
    const int arr_size = rows * cols;
    float *m_in, *m_out_a, *m_out_c, *m_out_d;
    m_in    = (float*) malloc(arr_size * sizeof(float));
    m_out_a = (float*) malloc(arr_size * sizeof(float));
    m_out_c = (float*) malloc(arr_size * sizeof(float));
    m_out_d = (float*) malloc(arr_size * sizeof(float));

    for (int i=0; i<(arr_size); i++){
        m_in[i] = i; // TODO random number
    }

    // initiate timing variable, keep results for validation
    unsigned long int elapsed_a, elapsed_c, elapsed_d;
    struct timeval t_start, t_end, t_diff;
    bool valid_c, valid_d; 
    
    // TASK 1 A)
    { 
        gettimeofday(&t_start, NULL); 

        matrix_accfun_cpu(cols, rows, m_in, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_a = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    }
    printf("Matrix accfun on size %d x %d on CPU runs in: %lu microsecs",cols, rows, elapsed_a);
    
    // TASK 1 B) OMITTED

    // TASK 1 C)
    { 
        gettimeofday(&t_start, NULL); 

        matrix_accfun_gpu<float>(cols, rows, m_in, m_out_c, false); 
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_c = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_c = validate<float>(arr_size, m_out_a, m_out_c);
    }
    printf("Matrix accfun on size %d x %d on GPU first impl runs in: %lu microsecs\n",cols, rows, elapsed_c);
    if (valid_c) printf("Implementation is VALID\n");
    else printf("Implementation is INVALID\n");


    // TASK 1 D)
    { 
        gettimeofday(&t_start, NULL); 

        matrix_accfun_gpu<float>(cols, rows, m_in, m_out_d, true);  
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_d = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_d = validate<float>(arr_size, m_out_a, m_out_d);
    }
    printf("Matrix accfun on size %d x %d on GPU rewrite runs in: %lu microsecs",cols, rows, elapsed_d);
    if (valid_d) printf("Implementation is VALID\n");
    else printf("Implementation is INVALID\n");

    // TODO print statistics, speedup difference and so on

    free(m_in);
    free(m_out_a);
    free(m_out_c);
    free(m_out_d);

}



void task_three(){
    // Dense Matrix-Matrix multiplication
    // 1a. implement serial version
    // 1b. bonus objective implement in OPENMP
    // 1c. implement naive  version
    // 1d. implement serial version

    // initiate data to transpose (dense matrix)
    const int res_size = M * N;
    float *m_in_a, *m_in_b, *m_out_a, *m_out_c, *m_out_d;
    m_in_a  = (float*) malloc(M * U * sizeof(float));
    m_in_b  = (float*) malloc(U * N * sizeof(float));
    m_out_a = (float*) malloc(res_size * sizeof(float));
    m_out_c = (float*) malloc(res_size * sizeof(float));
    m_out_d = (float*) malloc(res_size * sizeof(float));
    
    for (int i=0; i<(M*U); i++){
        m_in_a[i] = 0; // TODO random number
    }

    for (int i=0; i<(U*N); i++){
        m_in_b[i] = 0; // TODO random number
    }

    // initiate timing variable, keep results for validation
    unsigned long int elapsed_a, elapsed_c, elapsed_d;
    struct timeval t_start, t_end, t_diff;
    bool valid_c, valid_d; 
    
    // TASK 1 A)
    { 
        gettimeofday(&t_start, NULL); 

        matmult_cpu<float>(M, U, m_in_a, U, N, m_in_b, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_a = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    }
    printf("MatrixMult on (%dx%d) x (%d,%d) on CPU runs in: %lu microsecs",M,U,U,N, elapsed_a);
    
    // TASK 1 B) OMITTED

    // TASK 1 C)
    { 
        gettimeofday(&t_start, NULL); 

        matmult_gpu<float>(M, U, m_in_a, U, N, m_in_b, m_out_a, false);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_c = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_c = validate(res_size, m_out_a, m_out_c);
    }
    printf("MatrixMult on (%dx%d) x (%d,%d) on GPU runs in: %lu microsecs",M,U,U,N, elapsed_c);
    if (valid_c) printf("Implementation is VALID\n");
    else printf("Implementation is INVALID\n");


    // TASK 1 D)
    { 
        gettimeofday(&t_start, NULL); 

        matmult_gpu(M, U, m_in_a, U, N, m_in_b, m_out_a, true);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_d = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_d = validate(res_size, m_out_a, m_out_d);
    }
    printf("MatrixMult on (%dx%d) x (%d,%d) on CPU runs in: %lu microsecs",M,U,U,N, elapsed_d);
    if (valid_d) printf("Optimal Implementation is VALID\n");
    else printf("Optimal Implementation is INVALID\n");

    // TODO print statistics, speedup difference and so on

    free(m_in_a);
    free(m_in_b);
    free(m_out_a);
    free(m_out_c);
    free(m_out_d);

}

