#ifndef HOST_LIB
#define HOST_LIB

#include "devlib.cu.h"

#include <sys/time.h>
#include <time.h> 
#include <math.h>
#include <stdlib.h>

//#define BLOCK_SIZE 512
#define EPSILON 0.005

//****************************************************************************//
// DECLERATION OF ALL FUNCTIONS IMPLEMENTED IN THIS LIBRARY                   //
// ALL ARGUMENTS USED TO CALL THESE FUNCTIONS LIVE IN HOST MEMORY             //
// DURING PROCESSING, DEVICE MEMORY IS ALLOCATED AND HOST-DEVICE              //
// TRANSACTIONS ARE IMPLEMENTED AT THIS LEVEL                                 //
//                                                                            //
// DEVICE FUNCTIONS FROM DEVLIB.CU.H ARE CALLED FROM THIS LIBRARY             //
// WITH ARGUMENTS POINTING TO MEMORY ON THE DEVICE                            //
//****************************************************************************//



// MATRIX TRANSPOSITION (ASS3 TASK1)                                          //
template<class T> unsigned long int transpose_cpu( const unsigned int, const unsigned int, T*, T*);
template<class T> unsigned long int transpose_gpu( const unsigned int, const unsigned int, T*, T*, const unsigned char version=0, const unsigned char tile_size=0);

// MATRIX ACCUMULATION FUNCTION (ASS3 TASK2)                                  //
template<class T> unsigned long int matrix_accfun_cpu(const unsigned int, const unsigned int, T*, T*);       
template<class T> unsigned long int matrix_accfun_gpu(const unsigned int, const unsigned int, T*, T*, const unsigned char version=0, const unsigned int block_size=0 );   

// MATRIX MULTIPLICATION (ASS3 TASK3)                                         //
template<class T> unsigned long int matmult_cpu(const unsigned int, const unsigned int, T*, const unsigned int, const unsigned int, T*, T*);
template<class T> unsigned long int matmult_gpu(const unsigned int, const unsigned int, T*, const unsigned int, const unsigned int, T*, T*);

// (SEGMENTED) SCAN INCLUSIVE (WRAPPER TO PROVIDED FUNCTION)                  //
template<class OP, class T> void scanInc_gpu( const unsigned long, T*, T* );
template<class OP, class T> void sgmScanInc_gpu( const unsigned long, T*, int*, T* );

// MAXIMUM SEGMENT SUM (ASS2 PART1 TASK2)                                     //
int maxSegmentSum_gpu( const unsigned int, int*); 

// SPARSE MATRIX VECTOR MULTIPLICATION  (ASS2 PART1 TASK3)                    //
void spMatVecMult_gpu( const unsigned int, int*, int*, float*, float*, const unsigned int, float*);


// HELPER FUNCTIONS TO TIME AND VALIDATE (COULD BE MOVED OUT OF THIS LIBRARY) //
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);
template<class T> bool validate(const unsigned int, T*, T*, bool=false);
template<class T> T sum(const unsigned int, T* );
void matprint(const unsigned int, const unsigned int, int* );
void matprint(const unsigned int, const unsigned int, float* );
void matprint(const unsigned int, const unsigned int, double* );







/** MATRIX TRANSPOSITION (2D)                                                  *
 *  semantics: rows in outpu array = cols in input array and vice-versa        *
 *                                                                             *
 * The following functions hase same input and semantics,                      *
 * but differs in implementation                                               *
 *                                                                             *
 * rows_in    rows in input array (cols in output array)                       *
 * cols_in    cols in input array (rows in output array)                       *
 *                                                                             *
 * m_in       input matrix array                                               *
 * m_out      output matrix array                                              *
 *                                                                             *
 * (version)  char (small int) value to describe which implemented version     *
 *            to apply: std=0 is optimal   (GPU only)                          *
 *                                                                             *
 * (tile_size)integer which defines tile side dimension max 32                 *
 *            tile_size^2 becomes block_size in version 2                      *
 *                                                                             *
 */
/** SEQUENTIAL (ON CPU) **/
template<class T> unsigned long int transpose_cpu( const unsigned int rows_in, 
                                      const unsigned int cols_in, 
                                      T *m_in, 
                                      T *m_out
) {
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    for (int row=0; row<rows_in; row++){
        for (int col=0; col<cols_in; col++) {
            m_out[col*rows_in+row] = m_in[row*cols_in+col];
        }
    }
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    return elapsed;
}

/** PARALLEL (ON GPU) (additional 'naïve' argument) **/
template<class T> unsigned long int transpose_gpu( const unsigned int    rows_in, 
                                      const unsigned int    cols_in,
                                      T*                    h_in,        // host
                                      T*                    h_out,       // host
                                      const unsigned char   version,   
                                      const unsigned char   tile_size
){
    const unsigned int d_size = rows_in * cols_in;
    
    // allocate device arrays
    T *d_in, *d_out;
    cudaMalloc((void**)&d_in , d_size*sizeof(T));
    cudaMalloc((void**)&d_out, d_size*sizeof(T));

    // copy data to device
    cudaMemcpy( d_in, h_in, d_size*sizeof(T), cudaMemcpyHostToDevice);

    // solve problem using device (implementation in devlib.cu.h)
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    
    transpose<T>(rows_in, cols_in, d_in, d_out, version, tile_size);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    // copy result back from device
    cudaMemcpy( h_out, d_out, d_size*sizeof(T), cudaMemcpyDeviceToHost);

    // unallocate device arrays
    cudaFree(d_in);
    cudaFree(d_out);

    return elapsed;
}


/** MATRIX ACCUMULATION FUNCTION (ASS3 TASK2)                                  *
 *  semantics: for i from 0 to n-1  // outer loop                              *
 *               accum  = A[i,0] * A[i,0]                                      *
 *               B[i,0] = accum                                                *
 *               for j from 1 to 63 // inner loop                              *
 *                 tmpA   = A[i, j]                                            *
 *                 accum  = sqrt(accum) + tmpA * tmpA                          *
 *                 B[i,j] = accum                                              *
 *                                                                             *
 * The following functions hase same input and semantics,                      *
 * but differs in implementation                                               *
 *                                                                             *
 * rows_in    rows in input array (cols in output array)                       *
 * cols_in    cols in input array (rows in output array)                       *
 *                                                                             *
 * h_in       input matrix array   (host mem)                                  *
 * h_out      output matrix array  (host mem)                                  *
 *                                                                             *
 * (version)  char (small int) value to describe which version to be used      *
 *            std=0 is optimal version   (GPU only)                            *
 *                                                                             *
 */
template<class T> 
unsigned long int matrix_accfun_cpu( int rows_in, 
                        int cols_in, 
                        T* h_in, 
                        T* h_out
) {
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    T accum, tmp;
    for (int x=0 ; x<cols_in ; x++) {
        accum =  h_in[x] * h_in[x];
	h_out[x] = accum;
	for (int y=1 ; y<rows_in ; y++ ) {
	    tmp   = h_in[x + y * cols_in];
	    accum = sqrt(accum) + tmp * tmp;
	    h_out[x + y * cols_in] = accum;
	}
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    return elapsed;
}

template<class T> 
unsigned long int matrix_accfun_gpu( const unsigned int   rows_in, 
                        const unsigned int   cols_in, 
                        T*                   h_in, 
                        T*                   h_out, 
                        const unsigned char  version,
                        const unsigned int   block_size
) {    
    const unsigned int d_size = rows_in * cols_in;
    
    // allocate device arrays
    T *d_in, *d_out;
    cudaMalloc((void**)&d_in , d_size*sizeof(T));
    cudaMalloc((void**)&d_out, d_size*sizeof(T));

    // copy data to device
    cudaMemcpy( d_in, h_in, d_size * sizeof(T), cudaMemcpyHostToDevice);

    // solve problem using device (implementation in devlib.cu.h)
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    printf("entering devlib");
    matrix_accfun<T>(rows_in, cols_in, d_in, d_out, version, block_size);
    printf("exiting devlib");

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    // copy result back from device
    cudaMemcpy( h_out, d_out, d_size*sizeof(T), cudaMemcpyDeviceToHost);

    // unallocate device arrays
    cudaFree(d_in);
    cudaFree(d_out);

    return elapsed;
}




/** MATRIX MULTIPLICATION        (ASS3 TASK3)                                  *
 *  semantics: performs matrix multiplication on two input matrices and        *
 *             places result in output matrix                                  *
 *             caller is responsible for correct dimensionality of input       *
 *                                                                             *
 * The following functions hase same input and semantics,                      *
 * but differs in implementation                                               *
 *                                                                             *
 * rows_in_a  rows in input array (cols in output array)  M                    *
 * cols_in_a  cols in input array (rows in output array)  U                    *
 * h_in_a     input matrix array  (host mem)              MxU                  *
 *                                                                             *
 * rows_in_b  rows in input array (cols in output array)  U                    *
 * cols_in_b  cols in input array (rows in output array)  N                    *
 * h_in_b     input matrix array  (host mem)              UxN                  *
 *                                                                             *
 * h_out      output matrix array (host mem)              MxN                  *
 *                                                                             *
 * (opt)      boolean value to describe wether optimal implementation          *
 *            is requested... if on gpu,  standard is 'true'                   *
 *                                                                             *
 */
template<class T> 
unsigned long int matmult_cpu( const unsigned int rows_in_a, 
                  const unsigned int cols_in_a,
                  T*                 h_in_a, 
                  const unsigned int rows_in_b,
                  const unsigned int cols_in_b,
                  T*                 h_in_b,
                  T*                 h_out
) {
    if (cols_in_a != rows_in_b) {
        printf("matrix multiplication: input dimensions does not fit");
        return 0;
    }
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    //int m = rows_in_a;
    //int u = rows_in_b; // = cols_in_a
    //int n = cols_in_b;
    float tmp;
    for (int i=0 ; i<rows_in_a ; i++ ) {
        tmp = 0;
        for (int j=0 ; j<cols_in_b ; j++) {
	    for (int k=0 ; k<cols_in_a ; k++) {
	        tmp += h_in_a[i*rows_in_a+k] * h_in_b[k*rows_in_b+j];
	    }
            h_out[i*rows_in_a + j] = tmp;
	}
    }
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    return elapsed;
}

template<class T> 
unsigned long int matmult_gpu( const unsigned int rows_in_a, 
                  const unsigned int cols_in_a,
                  T*                 h_in_a, 
                  const unsigned int rows_in_b,
                  const unsigned int cols_in_b,
                  T*                 h_in_b,
                  T*                 h_out,
                  bool               opt
) {    
    const unsigned int d_size_a   = rows_in_a * cols_in_a;
    const unsigned int d_size_b   = rows_in_b * cols_in_b;
    const unsigned int d_size_out = rows_in_a * cols_in_b;
    const unsigned int block_size = BLOCK_SIZE;
    
    // allocate device arrays
    T *d_in_a, *d_in_b, *d_out;
    cudaMalloc((void**)&d_in_a , d_size_a*sizeof(T));
    cudaMalloc((void**)&d_in_b , d_size_b*sizeof(T));
    cudaMalloc((void**)&d_out, d_size_out*sizeof(T));

    // copy data to device
    cudaMemcpy( d_in_a, h_in_a, d_size_a*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy( d_in_b, h_in_b, d_size_b*sizeof(T), cudaMemcpyHostToDevice);

    // solve problem using device (implementation in devlib.cu.h)
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    if (opt) {
        matmult_opt<T>(block_size, rows_in_a, cols_in_a, d_in_a, rows_in_b, cols_in_b, d_in_b, d_out);
    } else {
        matmult<T>(block_size, rows_in_a, cols_in_a, d_in_a, rows_in_b, cols_in_b, d_in_b, d_out);
    }

    // copy result back from device
    cudaMemcpy( h_out, d_out, d_size_out*sizeof(T), cudaMemcpyDeviceToHost);

    // unallocate device arrays
    cudaFree(d_in_a);
    cudaFree(d_in_b);
    cudaFree(d_out);
    
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    return elapsed;
}

/******************************************************************************/
/* PREVIOUSLY IMPLEMENTED FUNCTIONS                                           */
/******************************************************************************/


/** (SEGMENTED) SCAN INCLUSIVE - TEMPLATE                       *
 *                                                              *
 * size       is the size of both the input and output arrays.  *
 * h_in       is the host array; it is supposably               *
 *                allocated and holds valid values (input).     *
 * (h_flags)  is the host flag array, in which !=0 indicates    *
 *                start of a segment.                           *
 * h_out      is the output hostarray                           *
 *                                                              *
 * OP         class denotes the associative binary operator     *
 *                and should have an implementation similar to  *
 *                `class Add' in ScanUtil.cu, i.e., exporting   *
 *                `identity' and `apply' functions.             *
 * T          denotes the type on which OP operates,            *
 *                e.g., float or int.                           *
 */
template<class OP, class T>
void scanInc_gpu(  const unsigned long size, 
                   T*                  h_in,  // host
                   T*                  h_out  // host
) {
    const unsigned int block_size = BLOCK_SIZE;

    // allocate gpu mem
    T *d_in, *d_out;
    cudaMalloc((void**)&d_in , size*sizeof(T));
    cudaMalloc((void**)&d_out, size*sizeof(T));
    
    // copy input from host mem to device mem
    cudaMemcpy( d_in, h_in, size*sizeof(T), cudaMemcpyHostToDevice);
    
    // call gpu scanInc
    scanInc(block_size, size, d_in, d_out);
    
    // copy result back to host mem
    cudaMemcpy( h_out, d_out, size*sizeof(T), cudaMemcpyDeviceToHost);
    
    // free dev mem
    cudaFree(d_in );
    cudaFree(d_out);

}

// SEGMENTED VERSION                                         //
template<class OP, class T>
void sgmScanInc_gpu( const unsigned long size,
                     T*                  h_in,    // host
                     int*                h_flags, // host
                     T*                  h_out    // host
) {
    const unsigned int block_size = BLOCK_SIZE;

    // allocate gpu mem
    T *d_in, *d_out;
    int *d_flags;
    cudaMalloc((void**)&d_in , size*sizeof(T));
    cudaMalloc((void**)&d_out, size*sizeof(T));
    cudaMalloc((void**)&d_flags, size*sizeof(int));
    
    // copy input from host mem to device mem
    cudaMemcpy( d_in, h_in, size*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy( d_flags, h_flags, size*sizeof(int), cudaMemcpyHostToDevice);
    
    // call gpu scanInc
    segmScanInc(block_size, size, d_in, d_flags, d_out);
    
    // copy result back to host mem
    cudaMemcpy( h_out, d_out, size*sizeof(T), cudaMemcpyDeviceToHost);
    
    // free dev mem
    cudaFree(d_in );
    cudaFree(d_out);
    cudaFree(d_flags);

}




/** MAXIMUM SEGMENT SUM                               *
 *                                                    *
 *  d_size      is total length of input array          *
 *  h_in      input array in which MSS is to be found *
 *                                                    *
 */
int maxSegmentSum_gpu( const unsigned int d_size,  
                       int*               h_in   // host 
) {
    const unsigned int block_size = BLOCK_SIZE;

    // allocate gpu mem
    int *d_in, res;
    cudaMalloc((void**)&d_in, d_size*sizeof(int));
    
    // copy input from host mem to device mem
    cudaMemcpy( d_in, h_in, d_size*sizeof(int), cudaMemcpyHostToDevice);
    
    // call gpu mss result is returned directly
    res = maxSegmentSum( block_size, d_size, d_in);
    
    // free dev mem
    cudaFree(d_in );

    return res;
}


/** SPARSE MATRIX VECTOR MULTIPLICATION                       *
 *                                                            *
 *  size       total number of entries in matrix              *
 *  h_flags    pointer to host array containing row-flags     *
 *  h_mat_idx  ptr to host array containing column-idxs       *
 *  h_mat_val  ptr to host array containing value in matrix   *
 *  h_vec_val  ptr to host array containing vector values     *
 *  out_size   number of rows in matrix (size of out array)   *
 *  h_out      ptr to host array in which result is outputted *
 */
void spMatVecMult_gpu( const unsigned int size,     
                       int*               h_flags,   // host
                       int*               h_mat_idx, // host
                       float*             h_mat_val, // host
                       float*             h_vec_val, // host
		       const unsigned int out_size,  
                       float*             h_out      // host
) {  
    const unsigned int block_size = BLOCK_SIZE;

    // allocate gpu mem
    int *d_flags, *d_mat_idx;
    float *d_mat_val, *d_vec_val, *d_out;
    
    cudaMalloc((void**)&d_flags,   size*sizeof(int));
    cudaMalloc((void**)&d_mat_idx, size*sizeof(int));
    cudaMalloc((void**)&d_mat_val, size*sizeof(float));
    cudaMalloc((void**)&d_vec_val, size*sizeof(float));
    cudaMalloc((void**)&d_out,     out_size*sizeof(float));
    
    // copy input from host mem to device mem
    cudaMemcpy( d_flags,   h_flags,   size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_mat_idx, h_mat_idx, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_mat_val, h_mat_val, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_vec_val, h_vec_val, size*sizeof(float), cudaMemcpyHostToDevice);
    
    // call gpu sparse Matrix-Vector multiplication function
    spMatVecMult(block_size, size, d_flags, d_mat_idx, d_mat_val, d_vec_val, d_out);
    
    // copy result back to host mem
    cudaMemcpy( h_out, d_out, out_size*sizeof(float), cudaMemcpyDeviceToHost);
    
    // free dev mem
    cudaFree(d_flags);
    cudaFree(d_mat_idx);
    cudaFree(d_mat_val);
    cudaFree(d_vec_val);
    cudaFree(d_out);

}



/******************************************************************************/
/* HELPER FUNCTIONS - SUPPORTING ACTUAL IMPLEMENTATION                        */
/******************************************************************************/


/** TIMEVALUE SUBTRACTOR                       *
 *  Helper function to calculate runtimes      *
 *                                             */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

/** ARRAY VALIDATION                                              *
 * tests that the first *size* elements of 2 arrays are identical *
 *                                                                *
 * size    is the number of elements to be tested                 *
 * arr_a   array to be tested                                     *
 * arr_b   against this array                                     *
 *                                                                *
 */
template<class T>
bool validate(const unsigned int size, T* arr_a, T* arr_b, bool verbose){
    bool success = true;
    T diff;
    for (int i=0; i < size; i++) {
        diff = abs(arr_a[i] - arr_b[i]);
        if ( diff > EPSILON ) {
            success = false;
            if (verbose) {
                printf("@%d: %f - %f = %f\n",i, arr_a[i], arr_b[i], diff);
            }
        }
    }
    return success;
}


/** ARRAY SUMMATION                                               *
 * calculates the sum of the following *size* numbers in array    *
 *                                                                *
 * size    is the number of elements to be added                  *
 * arr     array to be summarized                                 *
 *                                                                *
 */
template<class T> T sum(const unsigned int size, T* arr){
    T acc = 0;
    for (int i=0 ; i <size; i++) {
        acc += arr[i];
    }
    return acc;
}


/** MATRIX PRINTER
 *  pretty prints a matrix array ... 
 *
 */
void matprint(const unsigned int rows, const unsigned int cols, int* arr ){
    for (int i=0 ; i<rows ; i++){
        for (int j=0 ; j<cols ; j++){
            printf("%4d ", arr[i*cols+j]);
	}
	printf("\n");
    }
    printf("\n");
}

void matprint(const unsigned int rows, const unsigned int cols, float* arr ){
    for (int i=0 ; i<rows ; i++){
        for (int j=0 ; j<cols ; j++){
            printf("%6.2f ", arr[i*cols+j]);
	}
	printf("\n");
    }
    printf("\n");
}

void matprint(const unsigned int rows, const unsigned int cols, double* arr ){
    for (int i=0 ; i<rows ; i++){
        for (int j=0 ; j<cols ; j++){
            printf("%6.2f ", arr[i*cols+j]);
	}
	printf("\n");
    }
    printf("\n");
}



#endif // HOST_LIB
