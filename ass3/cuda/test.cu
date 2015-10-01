#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hostlib.cu.h"

// task one specific
#define M1 (1*1024)
#define N1 (1*1024)
// task two specific
#define M2 64 // As per assignment
#define N2 8
// task three specific
#define M3 10
#define U3 8
#define N3 10 




// declared with purpose of starting the file with its main function
void task_one(bool);
void task_two(bool);
void task_three();
int warmup(const unsigned int size);


int main(int argc, char** argv) {
    warmup(7624); // sole purpose is 'warming up GPU' so that timings get valid downstream.
    mprinter(4,6,1);
    task_one(false);
    task_two(true);
    //task_three();
    return 0;
}






void task_one(bool verbose){
    printf("\nASSIGNMENT3 TASK1: MATRIX TRANSPOSITION\n");
    // Transpose Matrix 
    // 1a. implement serial version
    // 1b. bonus objective implement in OPENMP
    // 1c. implement naive  version
    // 1d. implement serial version

    // initiate data to transpose (dense matrix)
    const unsigned int rows = M1;
    const unsigned int cols = N1;
    const unsigned int arr_size = rows * cols;

    float *m_in, *m_out_a, *m_out_c, *m_out_d;
    m_in    = (float*) malloc(arr_size * sizeof(float));
    m_out_a = (float*) malloc(arr_size * sizeof(float));
    m_out_c = (float*) malloc(arr_size * sizeof(float));
    m_out_d = (float*) malloc(arr_size * sizeof(float));

    for (int i=0; i<(arr_size); i++){
        m_in[i] = (float) i;
    }

    unsigned long int elapsed_a, elapsed_c, elapsed_d;
    bool valid_c, valid_d; 
    
    // TASK 1 A)
    elapsed_a = transpose_cpu<float>( rows, cols, m_in, m_out_a);
    printf("\nTranspose Matrix sized %d x %d on CPU ", cols, rows);
    printf("runs in: %lu microsecs\n", elapsed_a);
    
    // TASK 1 B) OMITTED

    // TASK 1 C)
    { 
        const unsigned char naive_version = 1;
        elapsed_c = transpose_gpu<float>( rows, cols, m_in, m_out_c, naive_version);
	valid_c   = validate<float>(arr_size, m_out_a, m_out_c);
    }
    printf("\nTranspose Matrix sized %d x %d on GPU naïvely ", cols, rows);
    printf("runs in: %lu microsecs\n", elapsed_c);
    if (valid_c) printf("Naïve implementation is VALID\n");
    else printf("Naïve implementation is INVALID\n");


    // TASK 1 D)
    { 
        const unsigned char opt_version = 2;
        elapsed_d = transpose_gpu<float>( rows, cols, m_in, m_out_d, opt_version);
	valid_d   = validate<float>(arr_size, m_out_a, m_out_d);
    }
    printf("\nTranspose Matrix sized %d x %d on GPU optimized ", cols, rows);
    printf("runs in: %lu microsecs\n", elapsed_d);
    if (valid_d) printf("Optimal implementation is VALID\n\n");
    else printf("Optimal implementation is INVALID\n\n");

    // TODO print statistics, speedup difference and so on

    if (verbose) {
        printf("\nInput matrix before transposition: \n");
        matprint(rows, cols, m_in);
        printf("Matrix after transposition by cpu: \n");
        matprint(cols, rows, m_out_a);
        printf("Matrix after transposition by gpu (naive): \n");
        matprint(cols, rows, m_out_c);
        printf("Matrix after transposition by gpu (opt): \n");
        matprint(cols, rows, m_out_d);
    }

    free(m_in);
    free(m_out_a);
    free(m_out_c);
    free(m_out_d);

    return;
}




void task_two(bool print){
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
        m_in[i] = 2; 
    }

    // initiate timing variable, keep results for validation
    unsigned long int elapsed_a, elapsed_c, elapsed_d;
    bool valid_c, valid_d; 
    
    // TASK 2 A)
    { 
        elapsed_a = matrix_accfun_cpu( rows, cols, m_in, m_out_a);
    }

    printf("\nMatrix accfun on size %d x %d on CPU ",cols, rows);
    printf("runs in: %lu microsecs\n", elapsed_a);
    

    // TASK 2 B) OMITTED

    // TASK 2 C)
    { 
	const unsigned char version = 1;
        elapsed_c = matrix_accfun_gpu<float>(rows, cols, m_in, m_out_c, version); 
	valid_c   = validate<float>(arr_size, m_out_a, m_out_c);
    }

    printf("\nMatrix accfun on size %d x %d on GPU first impl ",cols, rows);
    printf("runs in: %lu microsecs\n", elapsed_c);
    if (valid_c) printf("Implementation is VALID\n");
    else printf("Implementation is INVALID\n");


    // TASK 2 D)
    { 
	const unsigned char version = 2;
        elapsed_d = matrix_accfun_gpu<float>(rows, rows, m_in, m_out_d, version);  
	valid_d   = validate<float>(arr_size, m_out_a, m_out_d);
    }

    printf("\nMatrix accfun on size %d x %d on GPU rewrite ",cols, rows);
    printf("runs in: %lu microsecs\n", elapsed_d);
    if (valid_d) printf("Implementation is VALID\n\n");
    else printf("Implementation is INVALID\n\n");


    // TODO 
    // The modified program (CUDA transpositions included) has about two 
    // times the number of global memory accesses of the original program. 
    // Does it run faster or slower than the original, and by how much
    // (for a suitably large N)?

    if (print) {
        printf("Matrix accumulation function on following array: \n");
        matprint( rows, cols, m_in );
        printf("Matrix accumulation function by cpu (naïve): \n");
        matprint( rows, cols, m_out_a);
        printf("Matrix accumulation function by gpu (first version): \n");
        matprint( rows, cols, m_out_c);
        printf("Matrix accumulation function by gpu (second version): \n");
        matprint(rows, rows, m_out_d);
    }

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

    const unsigned int a_rows = M3;
    const unsigned int a_cols = U3;
    const unsigned int b_rows = U3;
    const unsigned int b_cols = N3;
 
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
    bool valid_c, valid_d; 
    
    // TASK 3 A)
    { 
        elapsed_a = matmult_cpu<float>(a_rows, a_cols, m_in_a, b_rows, b_cols, m_in_b, m_out_a);
    }

    printf("\nMatrixMult on (%dx%d) x (%d,%d) on CPU ",a_rows, a_cols, b_rows, b_cols);
    printf("runs in: %lu microsecs\n", elapsed_a);
    
    // TASK 3 B) OMITTED

    // TASK 3 C)
    { 
        elapsed_c = matmult_gpu<float>(a_rows, a_cols, m_in_a, b_rows, b_cols, m_in_b, m_out_a, false);
	valid_c   = validate(res_size, m_out_a, m_out_c);
    }
    printf("\nMatrixMult on (%dx%d) x (%d,%d) on GPU ",a_rows, a_cols, b_rows, b_cols);
    printf("runs in: %lu microsecs\n", elapsed_c);
    if (valid_c) printf("Implementation is VALID\n");
    else printf("Implementation is INVALID\n");


    // TASK 3 D)
    { 
        elapsed_d = matmult_gpu(a_rows, a_cols, m_in_a, b_rows, b_cols, m_in_b, m_out_a, true);
	valid_d   = validate(res_size, m_out_a, m_out_d);
    }
    printf("\nMatrixMult on (%dx%d) x (%d,%d) on CPU ",a_rows, a_cols, b_rows, b_cols);
    printf("runs in: %lu microsecs\n", elapsed_d);
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

int warmup(const unsigned int size){
    // performing max segment sum calculation for GPU warmup purpose
    int* h_in = (int*) malloc( size * sizeof(int));

    for(unsigned int i=0; i<size; i++) h_in[i] = 1;

    int res = maxSegmentSum_gpu( size, h_in );
    return res;
}
