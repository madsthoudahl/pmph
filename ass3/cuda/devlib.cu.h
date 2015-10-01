#ifndef DEV_LIB
#define DEV_LIB

#include "devkernels.cu.h"
#define TRANSPOSE_OPTIMAL_VERSION 2
#define MATRIX_ACCFUN_OPTIMAL_VERSION 1

/*******************************************************************************
 * DEVICE LIBRARY FUNCTIONS - UTILIZING KERNEL FUNCTIONS                       *
 ******************************************************************************/

// (SEGMENTED) SHIFT RIGHT                                                    //
template<class T> void ShiftRight( unsigned int, unsigned long, T*, T*, const T);
template<class T> void sgmShiftRight( unsigned int, unsigned long, T*, T*, T*, const T);

// (SEGMENTED) SCAN INCLUSIVE                                                 //
template<class OP, class T> 
void ScanInc( const unsigned int, const unsigned long, T*, T*);
template<class OP, class T> 
void sgmScanInc( const unsigned int, const unsigned long, T*, int*, T*);

// MAXIMUM SEGMENT SUM                                                        //
template<class T> T maxSegmentSum( unsigned int, unsigned int, T* );

// SPARSE MATRIX VECTOR MULTIPLICATION                                        //
void spMatVecMult( unsigned int, unsigned int, int*, int*, float*, float*, float* ); 


/*******************************************************************************
 * YET TO BE IMPLEMENTED                                                       *
 ******************************************************************************/

// MATRIX TRANSPOSITION         (ASS3 TASK1)                                  //
template<class T> void 
transpose( const unsigned int, const unsigned int, T*, T*, const unsigned char version=0, const unsigned char tile_size=TILE_SIZE);

// MATRIX ACCUMULATION FUNCTION (ASS3 TASK2)                                  //
template<class T> void 
matrix_accfun(const unsigned int, const unsigned int, T*, T*, const unsigned char version=0, const unsigned int block_size=0 );


// MATRIX MULTIPLICATION        (ASS3 TASK3)                                  //
template<class T> void 
matmult(const unsigned int, const unsigned int, const unsigned int, T*, const unsigned int, const unsigned int, T*, T* );
template<class T> void 
matmult_opt(const unsigned int, const unsigned int, const unsigned int, T*, const unsigned int, const unsigned int, T*, T* );







/*******************************************************************************
 * ACTUAL IMPLEMENTATIONS                                                      *
 ******************************************************************************/



/** SHIFT RIGHT  (SEGMENTED)                                       *
 *  Inserts a neutral value on initial position, and pushes every  *
 *  element one step right, discarding the last (in every segment) *
 *                                                                 *
 * block_size is the size of the cuda block (must be a multiple    *
 *                of 32 less than 1025)                            *
 * d_size     is the size of both the input and output arrays.     *
 * d_in       is the device array; it is supposably                *
 *                allocated and holds valid values (input).        *
 * (d_flags)  is an array describing where the segments start      *
 * d_out      is the output GPU array -- if you want               *
 *            its data on CPU needs to copy it back to host.       *
 *                                                                 *
 * OP         class denotes the associative binary operator        *
 *                 and should have an implementation similar to    *
 *                `class Add' in ScanUtil.cu, i.e., exporting      *
 *                `identity' and `apply' functions.                *
 * T          denotes the type on which OP operates,               *
 *                e.g., float or int.                              *
 *                                                                 */
template<class T>
void shiftRight( unsigned int  block_size,
                 unsigned long d_size, 
                 T*            d_in,  // device
                 T*            d_out, // device
                 const T       ne     // neutral element
) {
    unsigned int num_blocks;
    unsigned int sh_mem_size = block_size * 32; //sizeof(T);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    shiftRightByOne<T><<< num_blocks, block_size, sh_mem_size >>>
        (d_in, d_out, ne, d_size);
    cudaThreadSynchronize();
    
    return;
}

// SEGMENTED VERSION
template<class T>
void sgmShiftRight( unsigned int  block_size,
                    unsigned long d_size, 
                    T*            d_in,     // device
                    T*            flags_d,  // device
                    T*            d_out,    // device
                    const T       ne        // neutral element
) {
    unsigned int num_blocks;
    unsigned int sh_mem_size = block_size * 32; //sizeof(T);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    sgmShiftRightByOne<T><<< num_blocks, block_size, sh_mem_size >>>
        (d_in, flags_d, d_out, ne, d_size);
    cudaThreadSynchronize();
    
    return;
}




/** SCAN INCLUSIVE (SEGMENTED)                                   *
 *                                                               *
 * block_size is the size of the cuda block (must be a multiple  *
 *                of 32 less than 1025)                          *
 * d_size     is the size of both the input and output arrays    *
 * d_in       is the device array; it is supposably              *
 *                allocated and holds valid values (input).      *
 *(flags)     is the flag array, in which !=0 indicates          *
 *                start of a segment.                            *
 * d_out      is the output GPU array -- if you want             *
 *            its data on CPU needs to copy it back to host.     *
 *                                                               *
 * OP         class denotes the associative binary operator      *
 *                and should have an implementation similar to   *
 *                `class Add' in ScanUtil.cu, i.e., exporting    *
 *                `identity' and `apply' functions.              *
 * T          denotes the type on which OP operates,             *
 *                e.g., float or int.                            *
 *                                                               */
template<class OP, class T>
void scanInc(    unsigned int  block_size,
                 unsigned long d_size, 
                 T*            d_in,  // device
                 T*            d_out  // device
) {
    unsigned int num_blocks;
    unsigned int sh_mem_size = block_size * 32; //sizeof(T);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    scanIncKernel<OP,T><<< num_blocks, block_size, sh_mem_size >>>(d_in, d_out, d_size);
    cudaThreadSynchronize();
    
    if (block_size >= d_size) { return; }

    /**********************/
    /*** Recursive Case ***/
    /**********************/

    //   1. allocate new device input & output array of size num_blocks
    T *d_rec_in, *d_rec_out;
    cudaMalloc((void**)&d_rec_in , num_blocks*sizeof(T));
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T));

    unsigned int num_blocks_rec = ( (num_blocks % block_size) == 0 ) ?
                                  num_blocks / block_size     :
                                  num_blocks / block_size + 1 ; 

    //   2. copy in the end-of-block results of the previous scan 
    copyEndOfBlockKernel<T><<< num_blocks_rec, block_size >>>(d_out, d_rec_in, num_blocks);
    cudaThreadSynchronize();

    //   3. scan recursively the last elements of each CUDA block
    scanInc<OP,T>( block_size, num_blocks, d_rec_in, d_rec_out );

    //   4. distribute the the corresponding element of the 
    //      recursively scanned data to all elements of the
    //      corresponding original block
    distributeEndBlock<OP,T><<< num_blocks, block_size >>>(d_rec_out, d_out, d_size);
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
}

// SEGMENTED VERSION                                                          //
template<class OP, class T>
void sgmScanInc( const unsigned int  block_size,
                 const unsigned long d_size,
                 T*            d_in,  //device
                 int*          flags, //device
                 T*            d_out  //device
) {
    unsigned int num_blocks;
    //unsigned int val_sh_size = block_size * sizeof(T  );
    unsigned int flg_sh_size = block_size * sizeof(int);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    T     *d_rec_in;
    int   *f_rec_in;
    cudaMalloc((void**)&d_rec_in, num_blocks*sizeof(T  ));
    cudaMalloc((void**)&f_rec_in, num_blocks*sizeof(int));

    sgmScanIncKernel<OP,T> <<< num_blocks, block_size, 32*block_size >>>
                    (d_in, flags, d_out, f_rec_in, d_rec_in, d_size);
    cudaThreadSynchronize();
    //cudaError_t err = cudaThreadSynchronize();
    //if( err != cudaSuccess)
    //    printf("cudaThreadSynchronize error: %s\n", cudaGetErrorString(err));

    if (block_size >= d_size) { cudaFree(d_rec_in); cudaFree(f_rec_in); return; }

    //   1. allocate new device input & output array of size num_blocks
    T   *d_rec_out;
    int *f_inds;
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T   ));
    cudaMalloc((void**)&f_inds,    d_size    *sizeof(int ));

    //   2. recursive segmented scan on the last elements of each CUDA block
    sgmScanInc<OP,T>
                ( block_size, num_blocks, d_rec_in, f_rec_in, d_rec_out );

    //   3. create an index array that is non-zero for all elements
    //      that correspond to an open segment that crosses two blocks,
    //      and different than zero otherwise. This is implemented
    //      as a CUDA-block level inclusive scan on the flag array,
    //      i.e., the segment that start the block has zero-flags,
    //      which will be preserved by the inclusive scan. 
    scanIncKernel<Add<int>,int> <<< num_blocks, block_size, flg_sh_size >>>
                ( flags, f_inds, d_size );

    //   4. finally, accumulate the recursive result of segmented scan
    //      to the elements from the first segment of each block (if 
    //      segment is open).
    sgmDistributeEndBlock <OP,T> <<< num_blocks, block_size >>>
                ( d_rec_out, d_out, f_inds, d_size );
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
    cudaFree(f_rec_in );
    cudaFree(f_inds   );
}





/* MAXIMUM SEGMENT SUM                                                         *
 * calculates the sequence (segment) with the largest sum in the input array   *
 * and returns the sum                                                         *
 *                                                                             *
 * block_size    is the chosen device block size                               *
 * d_size        is the number of elements in the input array                  *
 * d_in          ptr to array in GPU memory to be investigated                 *
 *                                                                            */
int maxSegmentSum( unsigned int block_size, // block size chosen
                   unsigned int d_size,     // size of calculation
                   int*         d_in        // device memory pointer to input array
) {
    unsigned int num_blocks;
    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    //unsigned int mem_size_float = d_size * sizeof(float);
    unsigned int mem_size_myint = d_size * sizeof(MyInt4);

    MyInt4 *h_result = (MyInt4*) malloc(sizeof(MyInt4));
    MyInt4 *d_myint, *d_calc;

    cudaMalloc((void**)&d_myint, mem_size_myint);
    cudaMalloc((void**)&d_calc, mem_size_myint);

    // copy the values to a 4-tuple datastructure to support the calculations needed
    msspTrivialMap<<<num_blocks, block_size>>>(d_in, d_myint, d_size);

    // Use a Scan Inclusive with special MssOP operation on array
    scanInc< MsspOp, MyInt4 > ( block_size, d_size, d_myint, d_calc );

    // extract the last element of the calculation which hold the result 
    // Copy result back into host memory from device, or address pointed to will be wrong!
    cudaMemcpy( h_result, &d_calc[d_size-1], sizeof(MyInt4), cudaMemcpyDeviceToHost);
    int h_res = h_result[0].x;

    cudaFree(d_myint);
    cudaFree(d_calc);

    return h_res;
}





/* SPARSE MATRIX VECTOR MULTIPLICATION                                         *
 * calculates the output vector of such a multiplication and places it in d_out*
 *                                                                             *
 * block_size    is the chosen device block size                               *
 * d_size        is the number of elements in the input array                  *
 *                                                                             *
 *               pointers to arrays in DEVICE memory                           *
 * d_flags       flags describing rows in input matrix                         *
 * d_mat_idx     column for corresponding entry in d_mat_val                   *
 * d_mat_val     value of entry in input matrix                                *
 *                                                                             *
 * d_vec_val     input vector                                                  *
 *                                                                             *
 * d_out         output vector                                                 *
 *                                                                            */
void spMatVecMult(      unsigned int block_size,// size of each block used on the device 
                        unsigned int d_size,    // total number of entries in matrix
                        int*         d_flags,   // device
                        int*         d_mat_idx, // device
			float*      d_mat_val, // device
                        float*      d_vec_val, // device
		        float*      d_out      // device
) {
    unsigned int num_blocks;
    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;
    
    float*d_tmp_pairs, *d_tmp_sscan;
    int *d_tmp_idxs;
    cudaMalloc((void**)&d_tmp_pairs, sizeof(float) * d_size);
    cudaMalloc((void**)&d_tmp_sscan, sizeof(float) * d_size);
    cudaMalloc((void**)&d_tmp_idxs , sizeof(int) * d_size);
    
    printf("Performing sparse Matrix vector multiplication");
    
    // calculate array of products
    spMatVctMult_pairs<<<num_blocks, block_size>>>(d_mat_idx, d_mat_val, d_vec_val, d_size, d_tmp_pairs);

    // sum the products within their segment
    sgmScanInc< Add<float>,float> ( block_size, d_size, d_tmp_pairs, d_flags, d_tmp_sscan );
   
    // sum ( scan (+) 0 ) the flags to calculate indexes of results
    scanInc< Add<int>,int> ( block_size, d_size, d_flags, d_tmp_idxs );
    
    // write to the output array
    write_lastSgmElem<<< num_blocks, block_size >>>(d_tmp_sscan, d_tmp_idxs, d_flags, d_size, d_out);

    // clean up newly created arrays on device
    cudaFree(d_tmp_pairs);
    cudaFree(d_tmp_sscan);

}











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

        transpose_naive_kernel<<< grid_size, block_size >>> (rows_in, cols_in, d_in, d_out);

    } else if (ver==2) {

        block_size.x = t_size ;
        block_size.y = t_size ;
        
	grid_size.x = ( (cols_in % t_size == 0) ?
                         cols_in / t_size : 
                         cols_in / t_size + 1 );
        grid_size.y = ( (rows_in % t_size == 0) ?
                         rows_in / t_size :
                         rows_in / t_size + 1 );

        transpose_opt_kernel_old<<< grid_size, block_size >>> (rows_in, cols_in, d_in, d_out);

    } else {

        printf("devlib.cu.h: transpose: unknown function version, aborting");
        //block_size.y = ((TILE_SIZE) / 8);
        //transpose_opt_kernel_two<<< grid_size, block_size >>> (rows_in, cols_in, d_in, d_out);
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
    
    // if no version is chosen, use optimal version
    const char ver = (version==0) ? (MATRIX_ACCFUN_OPTIMAL_VERSION)  : version ; 
    
    // tile size argument is applied if available
   
    const unsigned int blck_size = (block_size==0) ? (BLOCK_SIZE) : block_size;
    if (blck_size>1024) {
        printf("devlib.cu.h: matrix_accfun: block size > 1024. aborting");
        return;
    }

    unsigned int num_blocks;
    num_blocks = ( (rows_in % block_size) == 0) ?
                    rows_in / block_size     :
                    rows_in / block_size + 1 ;

    if (ver==1) {
        
        mat_acc_kernel_first<T><<< num_blocks, block_size >>>(rows_in, cols_in, d_in, d_out);
    
    } else if (ver==2) {

        T* d_in_t, *d_out_t;
        cudaMalloc((void**) &d_in_t, rows_in*cols_in*sizeof(T) );
        cudaMalloc((void**) &d_out_t, rows_in*cols_in*sizeof(T) );

        transpose<T>( rows_in, cols_in, d_in, d_in_t );

        mat_acc_kernel_second<T><<< num_blocks, block_size >>>(rows_in, cols_in, d_in_t, d_out_t);
        printf("matrix_accfun_gpu_second not implemented in devlib.cu.h\n"); // TODO
        
	transpose<T>( cols_in, rows_in, d_out_t, d_out );
    
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
void matmult( const unsigned int block_size,
              const unsigned int rows_in_a,
              const unsigned int cols_in_a,
              T*                 d_in_a,
              const unsigned int rows_in_b,
              const unsigned int cols_in_b, 
              T*                 d_in_b, 
              T*                 d_out
) {
    // Implement a naïve CUDA version that straightforwardly implements the 
    // pseudo-code above. (Uses a two-dimensional kernel/grid corresponding 
    // to the two parallel outer loops.)

    printf("matmult_naive not implemented in devlib.cu.h\n"); // TODO
    return;
}

template<class T> 
void matmult_opt( const unsigned int block_size,
                  const unsigned int rows_in_a,
                  const unsigned int cols_in_a,
                  T*                 d_in_a,
                  const unsigned int rows_in_b,
                  const unsigned int cols_in_b, 
                  T*                 d_in_b, 
                  T*                 d_out
) {
    // Implement a CUDA optimized version that uses tiling in shared memory in
    // order to reduce the number of global-memory accesses by a factor of 
    // TILE-size – see lecture notes. (Uses a two-dimensional kernel/grid 
    // corresponding to the two parallel outer loops.)

    printf("matmult_opt not implemented in devlib.cu.h\n"); // TODO
    return;
}






#endif // DEV_LIB
