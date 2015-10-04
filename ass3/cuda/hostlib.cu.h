#ifndef HOST_LIB
#define HOST_LIB

#include "devlib.cu.h"

#include <sys/time.h>
#include <time.h> 
#include <math.h>
#include <stdlib.h>

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
template<class T> unsigned long int matmult_gpu(const unsigned int, const unsigned int, T*, const unsigned int, const unsigned int, T*, T*, const unsigned char version=0, const unsigned char tile_size=0);


// HELPER FUNCTIONS TO TIME AND VALIDATE (COULD BE MOVED OUT OF THIS LIBRARY) //
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);
template<class T> bool validate(const unsigned int, T*, T*, bool=false);
template<class T> bool mvalidate(const unsigned int, const unsigned int, T*, T*, bool=false);
template<class T> T sum(const unsigned int, T* );
void matprint(const unsigned int, const unsigned int, int* );
void matprint(const unsigned int, const unsigned int, float* );
void matprint(const unsigned int, const unsigned int, double* );
template<class T> void mprinter(const unsigned int rows_in, const unsigned int cols_in, T start);





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
    for (int i=0 ; i<rows_in ; i++) {
        accum =  h_in[i*cols_in] * h_in[i*cols_in];
	h_out[i*cols_in] = accum;
	for (int j=1 ; j<cols_in ; j++ ) {
	    tmp   = h_in[i * cols_in + j ];
	    accum = sqrt(accum) + tmp * tmp;
	    h_out[i * cols_in + j ] = accum;
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

    matrix_accfun<T>(rows_in, cols_in, d_in, d_out, version, block_size);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    // copy result back from device
    cudaMemcpy( h_out, d_out, d_size*sizeof(T), cudaMemcpyDeviceToHost);

    // unallocate device arrays
    cudaFree(d_out);
    cudaFree(d_in);

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
 * (version)  unsigned char value to describe which implementation             *
 *            is requested... if on gpu,  standard is 'optimal'                *
 *                                                                             *
 */
template<class T> 
unsigned long int matmult_cpu(
                  const unsigned int  rows_in_a, 
                  const unsigned int  cols_in_a,
                  T*                  h_in_a, 
                  const unsigned int  rows_in_b,
                  const unsigned int  cols_in_b,
                  T*                  h_in_b,
                  T*                  h_out
) {
    if (cols_in_a != rows_in_b) {
        printf("matrix multiplication: input dimensions does not fit");
        return 0;
    }
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    
    float tmp;
    for (int i=0 ; i<rows_in_a ; i++ ) {
        for (int j=0 ; j<cols_in_b ; j++) {
            tmp = 0;
	    for (int k=0 ; k<cols_in_a ; k++) {
	        tmp += h_in_a[i*cols_in_a+k] * h_in_b[k*cols_in_b+j];
	    }
            h_out[i*cols_in_b + j] = tmp;
	}
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    return elapsed;
}


template<class T> 
unsigned long int matmult_gpu( 
                  const unsigned int  rows_in_a, 
                  const unsigned int  cols_in_a,
                  T*                  h_in_a, 
                  const unsigned int  rows_in_b,
                  const unsigned int  cols_in_b,
                  T*                  h_in_b,
                  T*                  h_out,
                  const unsigned char version,
                  const unsigned char tile_size
) {    
    const unsigned int d_size_a   = rows_in_a * cols_in_a;
    const unsigned int d_size_b   = rows_in_b * cols_in_b;
    const unsigned int d_size_out = rows_in_a * cols_in_b;
    
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
 
    matmult<T>(rows_in_a, cols_in_a, d_in_a, rows_in_b, cols_in_b, d_in_b, d_out, version, tile_size);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    // copy result back from device
    cudaMemcpy( h_out, d_out, d_size_out*sizeof(T), cudaMemcpyDeviceToHost);

    // unallocate device arrays
    cudaFree(d_in_a);
    cudaFree(d_in_b);
    cudaFree(d_out);

    return elapsed;
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
                printf("A[%d]:%f  B[%d]:%f \n",i, arr_a[i], i, arr_b[i]);
            }
        }
    }
    return success;
}

template<class T> bool 
mvalidate(const unsigned int cols, const unsigned int rows, T* a, T* b, bool verbose){
    bool success = true;
    T diff;
    int e=0, strlen=50;
    char *errs;
    errs = (char*) malloc(cols*rows*strlen*sizeof(char));
    for (int i=0; i <cols*rows; i++) {
        diff = abs(a[i] - b[i]);
        if ( diff > EPSILON ) {
            success = false;
            if (verbose) {
                int x = i % cols;
                int y = i / cols;
                sprintf( &errs[e++*strlen], "A[%3d,%3d]:%8.3f  B[%3d,%3d]:%8.3f\n",x,y, a[i], x,y, b[i]);
            }
        }
    }
    if (verbose && e>0){
        printf("Matrix validation failed: log:\n");
        for (int i=0; i<e ; i++){
            printf("%s", &errs[i*strlen]);
        }
        free(errs);
        printf("Matrix validate log end\n");
    }
    return success;
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






template<class T> void mprinter(const unsigned int rows_in, const unsigned int cols_in, T start)
{
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


#endif // HOST_LIB
