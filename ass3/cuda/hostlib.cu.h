#ifndef HOST_LIB
#define HOST_LIB

#include "devlib.cu.h"

#include <sys/time.h>
#include <time.h> 

#define BLOCK_SIZE 512
#define EPSILON 0.00005

//****************************************************************************//
// DECLERATION OF ALL FUNCTIONS IMPLEMENTED IN THIS LIBRARY                   //
// ALL ARGUMENTS USED TO CALL THESE FUNCTIONS LIVE IN HOST MEMORY             //
// DURING PROCESSING, DEVICE MEMORY IS ALLOCATED AND HOST-DEVICE              //
// TRANSACTIONS ARE IMPLEMENTED AT THIS LEVEL                                 //
//                                                                            //
// DEVICE FUNCTIONS FROM DEVLIB.CU.H ARE CALLED FROM THIS LIBRARY             //
// WITH ARGUMENTS POINTING TO MEMORY ON THE DEVICE                            //
//****************************************************************************//


// HELPER FUNCTIONS TO TIME AND VALIDATE (COULD BE MOVED OUT OF THIS LIBRARY) //
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);
template<class T> bool validate(const unsigned int size, T* arr_a, T* arr_b);
template<class T> T sum(const unsigned int, T* );

// MATRIX TRANSPOSITION (ASS3 TASK1)                                          //
template<class T> void transpose_cpu( const unsigned int, const unsigned int, T*, T*);
template<class T> void transpose_gpu( const unsigned int, const unsigned int, T*, T*, bool);

// MATRIX ACCUMULATION FUNCTION (ASS3 TASK2)                                  //
template<class T> void matrix_accfun_cpu(const unsigned int, const unsigned int, T*, T*);       
template<class T> void matrix_accfun_gpu(const unsigned int, const unsigned int, T*, T*, bool);       

// MATRIX MULTIPLICATION (ASS3 TASK3)                                         //
template<class T> void matmult_cpu(const unsigned int, const unsigned int, T*, const unsigned int, const unsigned int, T*, T*);
template<class T> void matmult_gpu(const unsigned int, const unsigned int, T*, const unsigned int, const unsigned int, T*, T*);

// (SEGMENTED) SCAN INCLUSIVE (WRAPPER TO PROVIDED FUNCTION)                  //
template<class OP, class T> void scanInc_gpu( const unsigned long, T*, T* );
template<class OP, class T> void sgmScanInc_gpu( const unsigned long, T*, int*, T* );

// MAXIMUM SEGMENT SUM (ASS2 PART1 TASK2)                                     //
int maxSegmentSum_gpu( const unsigned int, int*); 

// SPARSE MATRIX VECTOR MULTIPLICATION  (ASS2 PART1 TASK3)                    //
void spMatVecMult_gpu( const unsigned int, int*, int*, float*, float*, const unsigned int, float*);








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
bool validate(const unsigned int size, T* arr_a, T* arr_b){
    bool success = true;
    for (int i=0; i < size; i++) {
        success &= ( abs( arr_a[i] - arr_b[i] ) < EPSILON );
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
    return acc
}



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
 * naive      boolean value to describe wether a naive or                      *
 *            optimal solution is requested                                    *
 *                                                                             *
 */
/** SEQUENTIAL (ON CPU) **/
template<class T> void transpose_cpu(int rows_in, int cols_in, T *m_in, T *m_out){
    for (int row=0; row<rows_in; row++){
        for (int col=0; col<cols_in; col++) {
            m_out[col*cols_in+row] = m_in[row*rows_in+col];
        }
    }
}

/** PARALLEL (ON GPU) (additional 'naïve' argument) **/
template<class T> void transpose_gpu( const unsigned int    rows_in, 
                                const unsigned int    cols_in,
                                T*                    h_in,        // host
                                T*                    h_out,       // host
		                bool                  naive=false  // optimal, unless specified
){
    const unsigned int d_size = rows_in * cols_in;
    const unsigned int block_size = BLOCK_SIZE;
    
    // allocate device arrays
    T *d_in, *d_out;
    cudaMalloc((void**)&d_in , d_size*sizeof(T));
    cudaMalloc((void**)&d_out, d_size*sizeof(T));

    // copy data to device
    cudaMemcpy( d_in, h_in, d_size*sizeof(T), cudaMemcpyHostToDevice);

    // solve problem using device (implementation in devlib.cu.h)
    if (naive) {
        transpose_naive<T>(block_size, rows_in, cols_in, d_in, d_out);
    } else {
        transpose_opt<T>(block_size, rows_in, cols_in, d_in, d_out);
    }

    // copy result back from device
    cudaMemcpy( h_out, d_out, d_size*sizeof(T), cudaMemcpyDeviceToHost);

    // unallocate device arrays
    cudaFree(d_in);
    cudaFree(d_out);
}


/** MATRIX ACCUMULATION FUNCTION (ASS3 TASK2)                                  *
 *  semantics: unknown for sure...                                             *
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
 * (second)   boolean value to describe wether second or                       *
 *            first solution is requested if on GPU                            *
 *                                                                             *
 */
template<class T> 
void matrix_accfun_cpu( int rows_in, 
                        int cols_in, 
                        T* h_in, 
                        T* h_out
) {
    printf("matrix_accfun_cpu not implemented in hostlib.cu.h"); // TODO
    return;
}

template<class T> 
void matrix_accfun_gpu( int rows_in, 
                        int cols_in, 
                        T* h_in, 
                        T* h_out, 
                        bool second=true
) {    
    const unsigned int d_size = rows_in * cols_in;
    const unsigned int block_size = BLOCK_SIZE;
    
    // allocate device arrays
    T *d_in, *d_out;
    cudaMalloc((void**)&d_in , d_size*sizeof(T));
    cudaMalloc((void**)&d_out, d_size*sizeof(T));

    // copy data to device
    cudaMemcpy( d_in, h_in, d_size * sizeof(T), cudaMemcpyHostToDevice);

    // solve problem using device (implementation in devlib.cu.h)
    if (second) {
        matrix_accfun_second<T>(block_size, rows_in, cols_in, d_in, d_out);
    } else {
        matrix_accfun_first<T>(block_size, rows_in, cols_in, d_in, d_out);
    }

    // copy result back from device
    cudaMemcpy( h_out, d_out, d_size*sizeof(T), cudaMemcpyDeviceToHost);

    // unallocate device arrays
    cudaFree(d_in);
    cudaFree(d_out);
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
void matmult_cpu( const unsigned int rows_in_a, 
                  const unsigned int cols_in_a,
                  T*                 h_in_a, 
                  const unsigned int rows_in_b,
                  const unsigned int cols_in_b,
                  T*                 h_in_b,
                  T*                 h_out
) {
    printf("matmult_cpu not implemented in hostlib.cu.h"); // TODO
    return;
}

template<class T> 
void matmult_gpu( const unsigned int rows_in_a, 
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
    
    return;
}




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
template<class T> T maxSegmentSum_gpu(  
                        const unsigned int d_size,  
                        T*                 h_in   // host 
) {
    const unsigned int block_size = BLOCK_SIZE;

    // allocate gpu mem
    T *d_in, res;
    cudaMalloc((void**)&d_in, d_size*sizeof(T));
    
    // copy input from host mem to device mem
    cudaMemcpy( d_in, h_in, d_size*sizeof(T), cudaMemcpyHostToDevice);
    
    // call gpu mss result is returned directly
    res = maxSegmentSum<T>( block_size, d_size, d_in);
    
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

    // calculate size of output // TODO make parallel implementation of 'sum'
    // const int out_size = sum<int>(size, h_flags);

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


#endif // HOST_LIB