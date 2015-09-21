#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ScanHost.cu.h"

#define EPS 0.0005
#define NUM_THREADS 898
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
        

        // copy device memory to host 
        // cudaMemcpy(h_out, d_out, mem_size_int, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("Wrapper: MaximumSegmentSum on GPU runs in: %lu microsecs\n", elapsed);

    // validation
    bool success = (result == NUM_THREADS);
    if (success) printf("\nWrapper: MSS +   VALID RESULT! %d\n",result);
    else         printf("\nWrapper: MSS + INVALID RESULT! %d should be %d\n",result, NUM_THREADS);
    printf("Wrapper: Largest Valid int is: %d\n",INT_MAX);

    // cleanup memory
    free(h_in );

    return result;
}
/*
// deprecated
int mssTest(){
    const unsigned int num_threads = NUM_THREADS;
    const unsigned int block_size  = BLOCK_SIZE;
    unsigned int mem_size_int = num_threads * sizeof(int);
    unsigned int mem_size_myint = num_threads * sizeof(MyInt4);

    int* h_in    = (int*) malloc(mem_size_int);
    int* h_out   = (int*) malloc(mem_size_int);

    { // init segments and flags
        for(unsigned int i=0; i<num_threads; i++) {
            h_in   [i] = 1; 
        }
    }

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 


    { // calling exclusive (segmented) scan
        int* d_in, *d_out;
        MyInt4* d_calc, *d_myint;
        cudaMalloc((void**)&d_in ,    mem_size_int);
        cudaMalloc((void**)&d_out ,   mem_size_int);
        cudaMalloc((void**)&d_myint,  mem_size_myint);
        cudaMalloc((void**)&d_calc,   mem_size_myint);
        //cudaMalloc((void**)&d_out,    sizeof(int));

        // copy host memory to device
        cudaMemcpy(d_in, h_in, mem_size_int, cudaMemcpyHostToDevice);

        { // execute kernels
            // copy the values to a 4-tuple datastructure to support the calculations needed
            createMyInt4array(block_size, num_threads, d_in, d_myint);
            // Use a Scan Inclusive with special MssOP operation on array
            scanInc< MsspOp, MyInt4 > ( block_size, num_threads, d_myint, d_calc );
	    // extract the last element of the calculation which hold the result
            extractLastAsInt( block_size, num_threads, d_calc, d_out ); // d_in should be d_out?
        }

        // copy device memory to host 
        cudaMemcpy(h_out, d_out, mem_size_int, cudaMemcpyDeviceToHost);

        // cleanup memory
        cudaFree(d_in );
        cudaFree(d_out);
        cudaFree(d_myint);
        cudaFree(d_calc);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("MaximumSegmentSum on GPU runs in: %lu microsecs\n", elapsed);

    // validation
    bool success = (h_out[0] == NUM_THREADS);
    if (success) printf("\nMSS +   VALID RESULT! %d\n",h_out[0]);
    else         printf("\nMSS + INVALID RESULT! %d should be %d\n",h_out[0], NUM_THREADS);
    printf("Largest Valid int is: %d\n",INT_MAX);

    //printf("mss cleanup %d \n", BLOCK_SIZE );
    // cleanup memory
    free(h_in );
    free(h_out );

    //printf("mss exiting %d \n", BLOCK_SIZE );
    return 0;
}
*/


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
    mssTest();
    msspTest();
}
