#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ScanHost.cu.h"

#define EPS 0.0005
#define NUM_THREADS 987624
#define BLOCK_SIZE 512


int msspTest(){
// using msspTest as a wrapper for one function in ScanHost.cu.h
    const unsigned int num_threads = NUM_THREADS;
    const unsigned int block_size  = BLOCK_SIZE;
    unsigned int mem_size_int = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size_int);
    int result;

    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1; 
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 


    { // calling exclusive (segmented) scan
        int* d_in;
        cudaMalloc((void**)&d_in ,    mem_size_int);

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size_int, cudaMemcpyHostToDevice);

        // test the wrapper
        result = maxSegmentSum ( block_size, num_threads, d_in );
        
        // cleanup memory
        cudaFree(d_in );
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("MaximumSegmentSum on GPU runs in: %lu microsecs", elapsed);

    // validation
    bool success = (result == NUM_THREADS);
    if (success) printf("\nMSS +   VALID RESULT! %d\n",result);
    else         printf("\nMSS + INVALID RESULT! %d should be %d\n",result, NUM_THREADS);

    // cleanup memory
    free(h_in );

    return result;
}


int scanExcTest(bool is_segmented) {
    const unsigned int num_threads = NUM_THREADS;
    const unsigned int block_size  = BLOCK_SIZE;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    int* h_out   = (int*) malloc(mem_size);
    int* flags_h = (int*) malloc(num_threads*sizeof(int));

    int sgm_size = 123;
    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1; 
            flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 


    { // calling exclusive (segmented) scan
        int* d_in;
        int* d_out;
        int* flags_d;
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        cudaMalloc((void**)&flags_d, num_threads*sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(flags_d, flags_h, num_threads*sizeof(int), cudaMemcpyHostToDevice);

        // execute kernel
        // REUSE d_in array by switching (d_in and d_out) arguments of (sgm)shiftRight...
        if(is_segmented) {
            sgmScanInc< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );
	    sgmShiftRight<int>( block_size, num_threads, d_out, flags_d, d_in, Add<int>::identity());
	}
        else {
            scanInc< Add<int>,int > ( block_size, num_threads, d_in, d_out );
	    shiftRight<int>( block_size, num_threads, d_out, d_in, Add<int>::identity());
	}

        // copy device memory to host - REMEMBER d_in holds result
        cudaMemcpy(h_out, d_in, mem_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(flags_d);
    }

    char segmented = ' ';
    if (is_segmented) segmented = 's';
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    //printf("Scan Exclusive on GPU runs in: %lu microsecs\n", elapsed);
    printf("%cScan Exclusive on GPU runs in: %lu microsecs", segmented, elapsed);

    // validation
    bool success = true;
    int  accum   = 0;
    if(is_segmented) {
        for(int i=0; i<num_threads; i++) {
            if (i % sgm_size == 0) accum  = 0;
            if ( abs(accum - h_out[i])>EPS ) { 
                success = false;
                printf("%cScan Exclusive Violation: %.1d should be %.1d\n", segmented, h_out[i], accum);
            }
            accum += 1;
        }        
    } else {
        for(int i=0; i<num_threads; i++) {
            if ( abs(accum - h_out[i])>EPS ) { 
                success = false;
                printf("%cScan Exclusive Violation: %.1d should be %.1d\n", segmented, h_out[i], accum);
            }
            accum += 1;
        }        
    }

    if(success) printf("\n%cScan Exclusive +   VALID RESULT!\n",segmented);
    else        printf("\n%cScan Exclusive + INVALID RESULT!\n",segmented);


    //printf("scanEx memcleanup %d \n", BLOCK_SIZE );
    // cleanup memory
    free(h_in );
    free(h_out);
    free(flags_h);
    //printf("scanEx exiting %d \n", BLOCK_SIZE );

    return 0;
}




int smvmTest() {
    const int    SIZE          = 10;
    const int    MAT_COLS      = 4;
    const int    VEC_LEN       = 4;
    const int    h_mat_flags[] = {1,0,1,0,0,1,0,0,1,0};
    const int    h_mat_idxs[]  = {0,1,0,1,2,1,2,3,2,3};
    const float  h_mat_vals[]  = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
    const float  h_vec_vals[]  = {2.0, 1.0, 0.0, 3.0};
    const float  res[]         = {3.0,0.0,-4.0,6.0}; // [(2*2-1*1), (-1*2+2*1), (-1*1-1*3),(2*3)]

    if (MAT_COLS != VEC_LEN) return -1;      // Matrix and vector not aligned

    const unsigned int num_threads = SIZE;
    const unsigned int block_size  = BLOCK_SIZE;
    unsigned int mem_size_float = num_threads * sizeof(float);
    unsigned int mem_size_result = MAT_COLS * sizeof(float);  
    unsigned int mem_size_int    = num_threads * sizeof(int);

    int    * h_in_mf  = (int*) malloc(mem_size_int);
    int    * h_in_mi  = (int*) malloc(mem_size_int);
    float  * h_in_mv  = (float*) malloc(mem_size_float);
    float  * h_in_vv  = (float*) malloc(mem_size_float);
    float  * h_out    = (float*) malloc(mem_size_result);

    for (int i=0; i<SIZE; i++) {
        h_in_mf[i] = h_mat_flags[i];
        h_in_mi[i] = h_mat_idxs[i];
        h_in_mv[i] = h_mat_vals[i];
        h_in_vv[i] = h_vec_vals[i];
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    {
        float  *d_in_mv, *d_in_vv, *d_out;
        int *d_in_mf, *d_in_mi;
        cudaMalloc((void**)&d_in_mf ,   mem_size_int);
        cudaMalloc((void**)&d_in_mi ,   mem_size_int);
        cudaMalloc((void**)&d_in_mv ,   mem_size_float);
        cudaMalloc((void**)&d_in_vv ,   mem_size_float);
        cudaMalloc((void**)&d_out   ,   mem_size_result);

        // copy host memory to device
        cudaMemcpy(d_in_mf, h_in_mf, mem_size_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_mi, h_in_mi, mem_size_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_mv, h_in_mv, mem_size_float, cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_vv, h_in_vv, mem_size_float, cudaMemcpyHostToDevice);

        // execute 'host' library function
	spMatVecMult(block_size, num_threads, d_in_mf, d_in_mi, d_in_mv, d_in_vv, d_out );

        // copy device memory to host
        cudaMemcpy(h_out, d_out, mem_size_result, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in_mf);
        cudaFree(d_in_mi);
        cudaFree(d_in_mv);
        cudaFree(d_in_vv);
        cudaFree(d_out);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("sparse Matrix-Vector multiplication on GPU runs in: %lu microsecs", elapsed);

    // validation
    bool success = true;
    for(int i=0; i<MAT_COLS; i++){
        success &= abs(h_out[i]-res[i])<EPS;
    }        
    
    printf("\nSparseMatrix Vector Multiplication Result:\n");
    printf("[ %f %f %f %f ] - calculation\n", res[0], res[1],  res[2], res[3]);
    printf("[ %f %f %f %f ] - correct result\n", h_out[0], h_out[1],  h_out[2], h_out[3]);

    if(success) printf("VALID RESULT");
    else        printf("INVALID RESULT");
    
    // cleanup memory
    free( h_in_mf);
    free( h_in_mi);
    free( h_in_mv);
    free( h_in_vv);
    free( h_out  ); 

    return 0;
}


int scanIncTest(bool is_segmented) {
    const unsigned int num_threads = NUM_THREADS;
    const unsigned int block_size  = BLOCK_SIZE;
    unsigned int mem_size = num_threads * sizeof(int);

    int* h_in    = (int*) malloc(mem_size);
    int* h_out   = (int*) malloc(mem_size);
    int* flags_h = (int*) malloc(num_threads*sizeof(int));

    int sgm_size = 123;
    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1; 
            flags_h[i] = (i % sgm_size == 0) ? 1 : 0;
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 


    { // calling inclusive (segmented) scan
        int* d_in;
        int* d_out;
        int* flags_d;
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        cudaMalloc((void**)&flags_d, num_threads*sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(flags_d, flags_h, num_threads*sizeof(int), cudaMemcpyHostToDevice);

        // execute kernel
        if(is_segmented)
            sgmScanInc< Add<int>,int > ( block_size, num_threads, d_in, flags_d, d_out );
        else
            scanInc< Add<int>,int > ( block_size, num_threads, d_in, d_out );

        // copy host memory to device
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(flags_d);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    char segmented = ' ';
    if (is_segmented) segmented = 's';
    printf("%cScan Inclusive on GPU runs in: %lu microsecs", segmented, elapsed);

    // validation
    bool success = true;
    int  accum   = 0;
    if(is_segmented) {
        for(int i=0; i<num_threads; i++) {
            if (i % sgm_size == 0) accum  = 0;
            accum += 1;
            
            if ( accum != h_out[i] ) { 
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }        
    } else {
        for(int i=0; i<num_threads; i++) {
            accum += 1;
 
            if ( accum != h_out[i] ) { 
                success = false;
                //printf("Scan Inclusive Violation: %.1d should be %.1d\n", h_out[i], accum);
            }
        }        
    }

    if(success) printf("\n%cScan Inclusive +   VALID RESULT!\n",segmented);
    else        printf("\n%cScan Inclusive + INVALID RESULT!\n",segmented);


    // cleanup memory
    free(h_in );
    free(h_out);
    free(flags_h);

    return 0;
}



int main(int argc, char** argv) {
    scanIncTest(true);
    scanIncTest(false);
    scanIncTest(true);
    scanExcTest(false);
    scanExcTest(true);
    msspTest();
    smvmTest();
}
