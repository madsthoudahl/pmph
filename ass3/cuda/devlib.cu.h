#ifndef DEV_LIB
#define DEV_LIB

#include "devkernels.cu.h"
#define TRANSPOSE_OPTIMAL_VERSION 2
#define MATRIX_ACCFUN_OPTIMAL_VERSION 2
#define MATRIX_MULTIPLICATION_OPTIMAL_VERSION 2

/*******************************************************************************
 * DEVICE LIBRARY FUNCTIONS - UTILIZING KERNEL FUNCTIONS                       *
 ******************************************************************************/


// MATRIX TRANSPOSITION         (ASS3 TASK1)                                  //
template<class T> void 
transpose( const unsigned int, const unsigned int, T*, T*, const unsigned char version=0, const unsigned char tile_size=0);

// MATRIX ACCUMULATION FUNCTION (ASS3 TASK2)                                  //
template<class T> void 
matrix_accfun(const unsigned int, const unsigned int, T*, T*, const unsigned char version=0, const unsigned int block_size=0 );

// MATRIX MULTIPLICATION        (ASS3 TASK3)                                  //
template<class T> void 
matmult(const unsigned int, const unsigned int, T*, const unsigned int, const unsigned int, T*, T*,const unsigned char version=0, const unsigned char tile_size=0 );




/*******************************************************************************
 * ACTUAL IMPLEMENTATIONS                                                      *
 ******************************************************************************/

/** MATRIX TRANSPOSITION                                                       *
 *  semantics: rows in outpu array = cols in input array and vice-versa        *
 *                                                                             *
 * block_size  size of blocks used in GPU computations div by 32 less than 1025*
 *                                                                             *
 * rows        number of rows in input array                                   *
 * cols        number of columns in input array                                *
 *             rows * cols = size of input matrix                              *
 *                                                                             *
 *             ptrs to arrays in device memory                                 *
 * d_in        input array, representing matrix                                *
 * d_out       output array                                                    *
 *                                                                             *
 * T           denotes type in entries of matrices, eg. int or floats         */
//  NAÏVE IMPLEMENTATION                                                      //
template<class T> void transpose( 
                                  const unsigned int  rows_in, 
                                  const unsigned int  cols_in,
                                  T*                  d_in,
                                  T*                  d_out,
                                  const unsigned char version,
                                  const unsigned char tile_size
){
    // Implement a “naive” transpose in CUDA, i.e., write a two-dimensional CUDA
    // kernel that exploits both N and M dimensions of parallelism and which 
    // performs the transposition much in the way shown in the pseudo-code
    
    // if no version is chosen, use optimal version
    const char ver = (version==0) ? (TRANSPOSE_OPTIMAL_VERSION)  : version ; 
    
    // tile size argument is applied if available
    unsigned int t_size = (tile_size==0) ? (TILE_SIZE) : tile_size ;
    if (t_size > 32) {
        printf("matrix transpose failing due to bad tile_size");
        return;
    } 

    dim3 block_size;
    dim3 grid_size;
    
    if (ver==1) {

        block_size.x = t_size ;
        block_size.y = t_size ;
        
	grid_size.x = ( (cols_in % t_size == 0) ? 
                         cols_in / t_size : 
                         cols_in / t_size + 1 );
        grid_size.y = ( (rows_in % t_size == 0) ?
                         rows_in / t_size :
                         rows_in / t_size + 1 );

        transpose_naive_kernel<T><<< grid_size, block_size >>> (rows_in, cols_in, d_in, d_out);
        cudaThreadSynchronize();

    } else if (ver==2) {

        block_size.x = t_size ;
        block_size.y = t_size ;
        
	grid_size.x = ( (cols_in % t_size == 0) ?
                         cols_in / t_size : 
                         cols_in / t_size + 1 );
        grid_size.y = ( (rows_in % t_size == 0) ?
                         rows_in / t_size :
                         rows_in / t_size + 1 );

        transpose_opt_kernel<T><<< grid_size, block_size >>> (rows_in, cols_in, d_in, d_out);
        cudaThreadSynchronize();

    } else {

        printf("devlib.cu.h: transpose: unknown function version, aborting");
    }
}








/** MATRIX ACCUMULATION FUNCTION (ASS3 TASK2)                                  *
 *  semantics: unknown for sure...                                             *
 *                                                                             *
 * The following functions hase same input and semantics,                      *
 * but differs in implementation                                               *
 *                                                                             *
 * block_size  is the size of the block used on the device                     *
 * rows_in     rows in input array (cols in output array)                      *
 * cols_in     cols in input array (rows in output array)                      *
 *                                                                             *
 * d_in        input matrix array   (device mem)                               *
 * d_out       output matrix array  (device mem)                               *
 *                                                                             *
 * (second)    boolean value to describe wether second or                      *
 *             first solution is requested if on GPU                           *
 *                                                                             */
template<class T>
void matrix_accfun(
                    const unsigned int  rows_in,
                    const unsigned int  cols_in,
                    T*                  d_in,
                    T*                  d_out,
                    const unsigned char version,
                    const unsigned int  block_size
) {
    
    //printf("entering matrix accfun in devlib.cu.h\n");
    // if no version is chosen, use optimal version
    unsigned int num_blocks;
    unsigned int d_size = rows_in * cols_in;
    const char ver = (version==0) ? (MATRIX_ACCFUN_OPTIMAL_VERSION)  : version ; 
    
    // blocksize argument is applied if available   
    const unsigned int blck_size = (block_size==0) ? (BLOCK_SIZE) : block_size;
    if (blck_size>1024) {
        printf("devlib.cu.h: matrix_accfun: block size > 1024. aborting\n");
        return;
    }

    if (ver==1) {
        
        num_blocks = ( (d_size % blck_size) == 0) ?
                        d_size / blck_size     :
                        d_size / blck_size + 1 ;

        mat_acc_first_kernel<T><<< num_blocks, blck_size >>>(rows_in, cols_in, d_in, d_out);
    
        cudaThreadSynchronize();

    } else if (ver==2) {

        num_blocks = ( (d_size % blck_size) == 0) ?
                        d_size / blck_size     :
                        d_size / blck_size + 1 ;

        T* d_in_t, *d_out_t;
        cudaMalloc((void**) &d_in_t,  d_size*sizeof(T) );
        cudaMalloc((void**) &d_out_t, d_size*sizeof(T) );

        transpose<T>( rows_in, cols_in, d_in, d_in_t );

        mat_acc_second_kernel<T><<< num_blocks, blck_size >>>( cols_in, rows_in, d_in_t, d_out_t);

	transpose<T>( cols_in, rows_in, d_out_t, d_out );
        cudaThreadSynchronize();
    
        cudaFree(d_in_t);
        cudaFree(d_out_t);

    } else {
        printf("devlib.cu.h: matrix_accfun: unknown function version, aborting");
    }
    return;
}








/** MATRIX MULTIPLICATION        (ASS3 TASK3)                                  *
 *  semantics: performs matrix multiplication on two input matrices and        *
 *             places result in output matrix                                  *
 *             caller is responsible for correct dimensionality of input       *
 *                                                                             *
 * The following functions hase same input and semantics,                      *
 * but differs in implementation                                               *
 *                                                                             *
 * block_size block size setting for device                                    *
 *                                                                             *
 * rows_in_a  rows in input array (cols in output array)                       *
 * cols_in_a  cols in input array (rows in output array)                       *
 * d_in_a     input matrix array  (device mem)                                 *
 *                                                                             *
 * rows_in_b  rows in input array (cols in output array)                       *
 * cols_in_b  cols in input array (rows in output array)                       *
 * d_in_b     input matrix array  (device mem)                                 *
 *                                                                             *
 * d_out      output matrix array (device mem)                                 *
 *                                                                             */
template<class T> 
void matmult( 
              const unsigned int  rows_in_a,
              const unsigned int  cols_in_a,
              T*                  d_in_a,
              const unsigned int  rows_in_b,
              const unsigned int  cols_in_b, 
              T*                  d_in_b, 
              T*                  d_out,
              const unsigned char version,
              const unsigned char tile_size
) {
    // Implement a naïve CUDA version that straightforwardly implements the 
    // pseudo-code above. (Uses a two-dimensional kernel/grid corresponding 
    // to the two parallel outer loops.)

    // if no version is chosen, use optimal version
    const char ver = (version==0) ? (MATRIX_MULTIPLICATION_OPTIMAL_VERSION)  : version ; 
    
    // tile size argument is applied if available
    unsigned int t_size = (tile_size==0) ? (TILE_SIZE) : tile_size ;
    if (t_size > 32) {
        printf("matrix transpose failing due to bad tile_size");
        return;
    } 

    dim3 block_size;
    dim3 grid_size;
    
    if (ver==1) {

        block_size.x = t_size ;
        block_size.y = t_size ;
        
        grid_size.x = ( (cols_in_b % t_size == 0) ?
                         cols_in_b / t_size :
                         cols_in_b / t_size + 1 );
	grid_size.y = ( (rows_in_a % t_size == 0) ? 
                         rows_in_a / t_size : 
                         rows_in_a / t_size + 1 );

        matmult_naive_kernel<T><<< grid_size, block_size >>>(rows_in_a, cols_in_a, cols_in_b, d_in_a, d_in_b, d_out);
    
        cudaThreadSynchronize();

    } else if (ver==2) {
        
        block_size.x = t_size ;
        block_size.y = t_size ;
        
        grid_size.x = ( (cols_in_b % t_size == 0) ?
                         cols_in_b / t_size :
                         cols_in_b / t_size + 1 );
	grid_size.y = ( (rows_in_a % t_size == 0) ? 
                         rows_in_a / t_size : 
                         rows_in_a / t_size + 1 );

        matmult_tile_kernel<T><<< grid_size, block_size >>>(rows_in_a, cols_in_a, cols_in_b, d_in_a, d_in_b, d_out);
    
        cudaThreadSynchronize();
    } else {
        printf("devlib.cu.h: matmult: unknown function version, aborting");
    }
    return;
}



#endif // DEV_LIB
