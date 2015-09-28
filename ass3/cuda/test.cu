#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hostlib.cu.h"

// in general
#define EPS 0.0005
#define BLOCK_SIZE 512
// for warmup and tests
#define NUM_THREADS 7624
#define MTEST 10
#define NTEST 10
// task one specific
#define ROWS_M 10 
#define COLS_N 8
// task two specific
#define M2 64 // As per assignment
#define N2 8
// task three specific
#define M 10
#define U 8
#define N 10 


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
template<class T> void mprinter(T);
void task_one();
void task_two();
void task_three();


int main(int argc, char** argv) {
    warmup(); // sole purpose is 'warming up GPU' so that timings get valid downstream.
    mprinter((float)1.0);
    mprinter(1);
    task_one();
    task_two();
    task_three();
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

template<class T>
void mprinter(T start){
    const unsigned int rows_in = MTEST;
    const unsigned int cols_in = NTEST;
    const unsigned int size = rows_in * cols_in;

    const unsigned int rows_out = cols_in;
    const unsigned int cols_out = rows_in;

    T* arr_in  = (T*) malloc( size * sizeof(T) );
    T* arr_out = (T*) malloc( size * sizeof(T) );
    T acc = start;
    for(unsigned int i=0; i<size; i++) {
        arr_in[i] = acc;
        acc += i;
    }

    transpose_cpu<T>( rows_in, cols_in, arr_in, arr_out);
    printf("\nMATRIX PRINTER:\n");
    printf("For testpurposes a matrix is printed, transposed and printed again.\n");
    matprint(rows_in, cols_in, arr_in);
    matprint(rows_out, cols_out, arr_out);

    free(arr_in);
    free(arr_out);
}

void task_one(){
    printf("\nASSIGNMENT3 TASK1: MATRIX TRANSPOSITION\n");
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
    printf("\nTranspose Matrix sized %d x %d on CPU runs in: %lu microsecs\n", cols, rows, elapsed_a);
    
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
    printf("\nTranspose Matrix sized %d x %d on GPU naïvely runs in: %lu microsecs\n", cols, rows, elapsed_c);
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
    printf("\nTranspose Matrix sized %d x %d on GPU optimized runs in: %lu microsecs\n", cols, rows, elapsed_d);
    if (valid_d) printf("Optimal implementation is VALID\n\n");
    else printf("Optimal implementation is INVALID\n\n");

    // TODO print statistics, speedup difference and so on

    free(m_in);
    free(m_out_a);
    free(m_out_c);
    free(m_out_d);

    return;
}




void task_two(){
    printf("\nASSIGNMENT3 TASK2: MATRIX TRANSPOSITION AS PREPROCESSING\n");
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
        m_in[i] = i * 1.0; // TODO random number
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
    printf("\nMatrix accfun on size %d x %d on CPU runs in: %lu microsecs\n",cols, rows, elapsed_a);
    
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
    printf("\nMatrix accfun on size %d x %d on GPU first impl runs in: %lu microsecs\n",cols, rows, elapsed_c);
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
    printf("\nMatrix accfun on size %d x %d on GPU rewrite runs in: %lu microsecs\n",cols, rows, elapsed_d);
    if (valid_d) printf("Implementation is VALID\n\n");
    else printf("Implementation is INVALID\n\n");

    // TODO 
    // The modified program (CUDA transpositions included) has about two 
    // times the number of global memory accesses of the original program. 
    // Does it run faster or slower than the original, and by how much
    // (for a suitably large N)?

    free(m_in);
    free(m_out_a);
    free(m_out_c);
    free(m_out_d);

}



void task_three(){
    printf("\nASSIGNMENT3 TASK3: MATRIX MULTIPLICATION\n");
    // Dense Matrix-Matrix multiplication
    // 1a. implement serial version
    // 1b. bonus objective implement in OPENMP
    // 1c. implement naive  version
    // 1d. implement serial version

    // initiate data to transpose (dense matrix)

    const unsigned int a_rows = M;
    const unsigned int a_cols = U;
    const unsigned int b_rows = U;
    const unsigned int b_cols = N;
 
    const unsigned int a_size   = a_rows * a_cols;
    const unsigned int b_size   = b_rows * b_cols;
    const unsigned int res_size = a_rows * b_cols;
    float *m_in_a, *m_in_b, *m_out_a, *m_out_c, *m_out_d;
    m_in_a  = (float*) malloc(  a_size * sizeof(float));
    m_in_b  = (float*) malloc(  b_size * sizeof(float));
    m_out_a = (float*) malloc(res_size * sizeof(float));
    m_out_c = (float*) malloc(res_size * sizeof(float));
    m_out_d = (float*) malloc(res_size * sizeof(float));
    
    for (int i=0; i<(a_size); i++){
        m_in_a[i] = i; // TODO random number
    }

    for (int i=0; i<(b_size); i++){
        m_in_b[i] = i; // TODO random number
    }

    // initiate timing variable, keep results for validation
    unsigned long int elapsed_a, elapsed_c, elapsed_d;
    struct timeval t_start, t_end, t_diff;
    bool valid_c, valid_d; 
    
    // TASK 1 A)
    { 
        gettimeofday(&t_start, NULL); 

        matmult_cpu<float>(a_rows, a_cols, m_in_a, b_rows, b_cols, m_in_b, m_out_a);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_a = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    }
    printf("\nMatrixMult on (%dx%d) x (%d,%d) on CPU runs in: %lu microsecs\n",a_rows, a_cols, b_rows, b_cols, elapsed_a);
    
    // TASK 1 B) OMITTED

    // TASK 1 C)
    { 
        gettimeofday(&t_start, NULL); 

        matmult_gpu<float>(a_rows, a_cols, m_in_a, b_rows, b_cols, m_in_b, m_out_a, false);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_c = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_c = validate(res_size, m_out_a, m_out_c);
    }
    printf("\nMatrixMult on (%dx%d) x (%d,%d) on GPU runs in: %lu microsecs\n",a_rows, a_cols, b_rows, b_cols, elapsed_c);
    if (valid_c) printf("Implementation is VALID\n");
    else printf("Implementation is INVALID\n");


    // TASK 1 D)
    { 
        gettimeofday(&t_start, NULL); 

        matmult_gpu(a_rows, a_cols, m_in_a, b_rows, b_cols, m_in_b, m_out_a, true);
    
        gettimeofday(&t_end, NULL); 
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_d = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
	valid_d = validate(res_size, m_out_a, m_out_d);
    }
    printf("\nMatrixMult on (%dx%d) x (%d,%d) on CPU runs in: %lu microsecs\n",a_rows, a_cols, b_rows, b_cols, elapsed_d);
    if (valid_d) printf("Optimal Implementation is VALID\n\n");
    else printf("Optimal Implementation is INVALID\n\n");

    // TODO 
    // Measure and compare the various running times. 
    // How many GFlops does the naïve and optimized CUDA versions achieve?

    free(m_in_a);
    free(m_in_b);
    free(m_out_a);
    free(m_out_c);
    free(m_out_d);

}

