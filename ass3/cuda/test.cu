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
#define M2 (6*64*1024)  // N in assignment
#define N2 64 // M=64 in assignment
// task three specific
#define M3 8
#define U3 5
#define N3 10 




// declared with purpose of starting the file with its main function
void task_one(bool);
void task_two(bool);
void task_three(bool);
int warmup(const unsigned int size);


int main(int argc, char** argv) {
    warmup(7624); // sole purpose is 'warming up GPU' so that timings get valid downstream.
    task_one(false);
    //task_two(false);
    task_three(true);
    return 0;
}






void task_one(bool verbose){
    printf("\n\nASSIGNMENT3 TASK1: MATRIX TRANSPOSITION\n");
    // Transpose Matrix 
    // 1a. implement serial version
    // 1b. bonus objective implement in OPENMP
    // 1c. implement naive  version
    // 1d. implement serial version

    // initiate data to transpose (dense matrix)
    const unsigned int rows = M1;
    const unsigned int cols = N1;
    const char s_valid[10]   = "--  VALID";
    const char s_invalid[10] = "--INVALID";
    const unsigned int arr_size = rows * cols;

    float *m_in, *m_out_a, *m_out_c, *m_out_d;
    m_in    = (float*) malloc(arr_size * sizeof(float));
    m_out_a = (float*) malloc(arr_size * sizeof(float));
    m_out_c = (float*) malloc(arr_size * sizeof(float));
    m_out_d = (float*) malloc(arr_size * sizeof(float));

    // INITIALIZE MATRIX (ARRAY) TO WORK ON
    for (int i=0; i<(arr_size); i++){
        m_in[i] = (float) i;
    }

    unsigned long int elapsed_a, elapsed_c, elapsed_d;
    bool valid_c, valid_d; 
    
    // TASK 1 A)
    {
        elapsed_a = transpose_cpu<float>( rows, cols, m_in, m_out_a);
    }
    
    // TASK 1 B) OMITTED

    // TASK 1 C)
    { 
        const unsigned char naive_version = 1;
        elapsed_c = transpose_gpu<float>( rows, cols, m_in, m_out_c, naive_version);
	valid_c   = validate<float>(arr_size, m_out_a, m_out_c);
    }

    // TASK 1 D)
    { 
        const unsigned char opt_version = 2;
        elapsed_d = transpose_gpu<float>( rows, cols, m_in, m_out_d, opt_version);
	valid_d   = validate<float>(arr_size, m_out_a, m_out_d);
    }

    // print statistics
    const double gpu_speedup = ((double) elapsed_c) / ((double) elapsed_d + 1);
    const double cpu_speedup = ((double) elapsed_a) / ((double) elapsed_d + 1);
    const unsigned int mops = 2 * rows * cols;
    float gmops_a = (float) mops / ( elapsed_a * 1000);
    float gmops_c = (float) mops / ( elapsed_c * 1000);
    float gmops_d = (float) mops / ( elapsed_d * 1000);

    printf("\nTranspose Matrix sized %d x %d running times\n", cols, rows);
    printf("CPU:           %10lu microsecs. \n", elapsed_a);
    printf("GPU naïve:     %10lu microsecs. %s\n", elapsed_c, (valid_c ? s_valid: s_invalid));
    printf("GPU optimized: %10lu microsecs. %s\n", elapsed_d, (valid_d ? s_valid: s_invalid));

    printf("\nGiga MemoryOPerations per second:\n");
    printf("CPU:           %10.3f Gmop/s.\n", gmops_a);
    printf("GPU naïve:     %10.3f Gmop/s.\n", gmops_c);
    printf("GPU optimized: %10.3f Gmop/s.\n", gmops_d);

    printf("\nThis is a speedup of %7.2f, for tile and coalesced mem accesses on GPU.\n",gpu_speedup);
    printf("... and a speedup of %7.2f, for GPU opt compared to CPU.\n",cpu_speedup);

    if (verbose) { // also print output matrices
        printf("\nInput matrix before transposition: \n");
        matprint(rows, cols, m_in);
        printf("Matrix after transposition by cpu: \n");
        matprint(cols, rows, m_out_a);
        printf("Matrix after transposition by gpu (naive): \n");
        matprint(cols, rows, m_out_c);
        printf("Matrix after transposition by gpu (opt): \n");
        matprint(cols, rows, m_out_d);
    }

    // unallocate memory again
    free(m_in);
    free(m_out_a);
    free(m_out_c);
    free(m_out_d);

    return;
}




void task_two(bool verbose){
    printf("\n\nASSIGNMENT3 TASK2: MATRIX TRANSPOSITION AS PREPROCESSING\n");
    // Matrix Transposition during or as a pre-computatiion

    const int rows = M2;
    const int cols = N2;
    const char s_valid[10]   = "--  VALID";
    const char s_invalid[10] = "--INVALID";
    const int arr_size = rows * cols;
    float *m_in, *m_out_a, *m_out_c, *m_out_d;
    m_in    = (float*) malloc(arr_size * sizeof(float));
    m_out_a = (float*) malloc(arr_size * sizeof(float));
    m_out_c = (float*) malloc(arr_size * sizeof(float));
    m_out_d = (float*) malloc(arr_size * sizeof(float));
    unsigned long int elapsed_a, elapsed_c, elapsed_d;
    bool valid_c, valid_d; 

    // INITIALIZE MATRIX (ARRAY) TO WORK ON
    for (int i=0; i<(arr_size); i++){
        m_in[i] =  2.0; 
    }

    // TASK 2 A)
    { 
        elapsed_a = matrix_accfun_cpu( rows, cols, m_in, m_out_a);
    }

    // TASK 2 B) OMITTED

    // TASK 2 C)
    { 
	const unsigned char version = 1;
        elapsed_c = matrix_accfun_gpu<float>(rows, cols, m_in, m_out_c, version); 
	valid_c   = validate<float>(arr_size, m_out_a, m_out_c);
    }

    // TASK 2 D)
    { 
	const unsigned char version = 2;
        elapsed_d = matrix_accfun_gpu<float>(rows, cols, m_in, m_out_d, version);  
	valid_d   = validate<float>(arr_size, m_out_a, m_out_d);
    }

    // The modified program (CUDA transpositions included) has about two 
    // times the number of global memory accesses of the original program. 
    // Does it run faster or slower than the original, and by how much
    // (for a suitably large N)?
    
    // print statistics
    const double gpu_speedup = ((double) elapsed_c) / ((double) elapsed_d + 1);
    const double cpu_speedup = ((double) elapsed_a) / ((double) elapsed_d + 1);

    printf("\nMatrix accfun on size %d x %d. Running times:\n", rows, cols);
    printf("CPU:           %10lu microsecs. \n", elapsed_a);
    printf("GPU first:     %10lu microsecs. %s\n", elapsed_c, (valid_c ? s_valid: s_invalid));
    printf("GPU second:    %10lu microsecs. %s\n", elapsed_d, (valid_d ? s_valid: s_invalid));

    printf("\nThis is a speedup of %7.2f, for second compared to first on GPU.\n",gpu_speedup);
    printf("... and a speedup of %7.2f, for GPU second compared to CPU.\n",cpu_speedup);
    
    if (verbose) {
        printf("Matrix accumulation function on following array: \n");
        matprint( rows, cols, m_in );
        printf("Matrix accumulation function by cpu (naïve): \n");
        matprint( rows, cols, m_out_a);
        printf("Matrix accumulation function by gpu (first version): \n");
        matprint( rows, cols, m_out_c);
        printf("Matrix accumulation function by gpu (second version): \n");
        matprint( rows, cols, m_out_d);
    }

    free(m_in);
    free(m_out_a);
    free(m_out_c);
    free(m_out_d);

}



void task_three(bool verbose){
    printf("\n\nASSIGNMENT3 TASK3: MATRIX MULTIPLICATION\n");

    const unsigned int a_rows = M3;
    const unsigned int a_cols = U3;
    const unsigned int b_rows = U3;
    const unsigned int b_cols = M3; //N3;
    const char s_valid[10]   = "--  VALID";
    const char s_invalid[10] = "--INVALID";
    const unsigned int a_size   = a_rows * a_cols;
    const unsigned int b_size   = b_rows * b_cols;
    const unsigned int res_size = a_rows * b_cols;
    float *m_in_a, *m_in_b, *m_out_a, *m_out_c, *m_out_d;
    m_in_a  = (float*) malloc(  a_size * sizeof(float));
    m_in_b  = (float*) malloc(  b_size * sizeof(float));
    m_out_a = (float*) malloc(res_size * sizeof(float));
    m_out_c = (float*) malloc(res_size * sizeof(float));
    m_out_d = (float*) malloc(res_size * sizeof(float));
    
    // INITIALIZE MATRICES (ARRAYS) TO WORK ON
    for (int i=0; i<(a_rows); i++){
        for (int j=0; j<(a_cols); j++){
            m_in_a[i*a_cols+j] = 0;
            if (abs(i-j)<2) {
                m_in_a[i*a_cols+j] = (i==j) ? 3.0 : -1.0 ;
            }
        }
    }

    transpose_gpu<float>( a_rows, a_cols, m_in_a, m_in_b );

    // initiate timing variable, keep results for validation
    unsigned long int elapsed_a, elapsed_c, elapsed_d;
    bool valid_c, valid_d; 
    
    // TASK 3 A)
    { 
        elapsed_a = matmult_cpu<float>(a_rows, a_cols, m_in_a, b_rows, b_cols, m_in_b, m_out_a);
    }
    
    // TASK 3 B) OMITTED

    // TASK 3 C)
    { 
	const unsigned char version = 1;
        elapsed_c = matmult_gpu<float>(a_rows, a_cols, m_in_a, b_rows, b_cols, m_in_b, m_out_c, version);
	valid_c   = validate(res_size, m_out_a, m_out_c);
    }

    // TASK 3 D)
    { 
	const unsigned char version = 2;
        elapsed_d = matmult_gpu(a_rows, a_cols, m_in_a, b_rows, b_cols, m_in_b, m_out_d, version);
	valid_d   = validate(res_size, m_out_a, m_out_d);
    }

    // Measure and compare the various running times. 
    // How many GFlops does the naïve and optimized CUDA versions achieve?
    const unsigned int flops = a_rows * (a_cols+b_rows) * b_cols;
    float gflops_a = (float) flops / (elapsed_a * 1000);
    float gflops_c = (float) flops / (elapsed_c * 1000);
    float gflops_d = (float) flops / (elapsed_d * 1000);

    // print statistics
    const double gpu_speedup = ((double) elapsed_c) / ((double) elapsed_d + 1);
    const double cpu_speedup = ((double) elapsed_a) / ((double) elapsed_d + 1);

    printf("\nMatrix Multiplication on (%dx%d) x (%d,%d). Timings:\n",a_rows, a_cols, b_rows, b_cols);
    printf("CPU:           %10lu microsecs. \n", elapsed_a);
    printf("GPU naïve:     %10lu microsecs. %s\n", elapsed_c, (valid_c ? s_valid: s_invalid));
    printf("GPU optimized: %10lu microsecs. %s\n", elapsed_d, (valid_d ? s_valid: s_invalid));

    printf("\nGiga FLoatingpointOPerations per second:\n");
    printf("CPU:           %10.3f Gflop/s.\n", gflops_a);
    printf("GPU naïve:     %10.3f Gflop/s.\n", gflops_c);
    printf("GPU optimized: %10.3f Gflop/s.\n", gflops_d);

    printf("\nThis is a speedup of %7.2f, for second compared to first on GPU.\n",gpu_speedup);
    printf("... and a speedup of %7.2f, for GPU second compared to CPU.\n\n",cpu_speedup);
    

    if (verbose) {
        printf("Matrix multiplication on following array: \n");
        //matprint( a_rows, a_cols, m_in_a );
        //matprint( b_rows, b_cols, m_in_b );
        printf("Matrix multiplication by cpu (naïve): \n");
        matprint( a_rows, b_cols, m_out_a);
        printf("Matrix multiplication by gpu (naïve): \n");
        matprint( a_rows, b_cols, m_out_c);
        printf("Matrix multiplication by gpu (optimized): \n");
        matprint( a_rows, b_cols, m_out_d);
    }

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
