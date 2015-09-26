#ifndef HOST_LIB
#define HOST_LIB

#include "devlib.cu.h"

#include <sys/time.h>
#include <time.h> 

#define BLOCK_SIZE 512


// declaration of functions used in main
// ALL should be moved to hostlib and implemented there
bool validate(int size, float* ground_truth, float* same);
 
int transpose_cpu(int rows_in, int cols_in, float *m_in, float *m_out);
int transpose_gpu_naive(int rows_in, int cols_in, float *m_in, float *m_out); // TODO
int transpose_gpu(int rows_in, int cols_in, float *m_in, float *m_out);       // TODO

int matrix_accfun_cpu(int rows_in, int cols_in, float* m_in, float* m_out_a);        // TODO 
int matrix_accfun_gpu_first(int rows_in, int cols_in, float* m_in, float* m_out_a);  // TODO
int matrix_accfun_gpu_second(int rows_in, int cols_in, float* m_in, float* m_out_a); // TODO

int matmult_cpu(int M, int U, float* m_in_a, int U, int N, float* m_in_b, float* m_out_a);     // TODO
int matmult_gpu(int M, int U, float* m_in_a, int U, int N, float* m_in_b, float* m_out_a);     // TODO
int matmult_gpu_opt(int M, int U, float* m_in_a, int U, int N, float* m_in_b, float* m_out_a); // TODO



int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

/**
 * tests that the first *size* elements of 2 arrays are identical
 *
 * size    is the number of elements to be tested
 * arr_a   array to be tested 
 * arr_b   against this array
 *
 */
bool validate(int size, float* arr_a, float* arr_b){
    bool success = true;
    for (int i=0; i < size; i++) {
        success &= ( abs( arr_a[i] - arr_b[i] ) < EPSILON );
    }
    return success;
}


/**
 *
 * The following functions hase same input and semantics, but differs in implementation
 * All transposes an input array 
 *
 * rows_in    rows in input array (cols in output array)
 * cols_in    cols in input array (rows in output array)
 *
 * m_in       input matrix array
 * m_out      output matrix array
 *
 */
int transpose_cpu(int rows_in, int cols_in, float *m_in, float *m_out){
    for (row=0; row<rows_in; row++){
        for (col=0; col<cols_in; col++) {
            m_out[col*cols_in+row] = m_in[row*rows_in+col];
        }
    }
}

int transpose_gpu_naive(int rows_in, int cols_in, float *h_in, float *h_out){
    const unsigned int d_size = rows_in * cols_in;
    const unsigned int block_size = BLOCK_SIZE;
    unsigned int num_blocks = ( (d_size % block_size) == 0) ?
                                 d_size / block_size     :
                                 d_size / block_size + 1 ;
    
    unsigned int sh_mem_size = block_size * 32; //sizeof(T);
    
    // allocate device arrays
    float* d_ini, d_out;
    cudaMalloc((void**)&d_in , d_size*sizeof(float));
    cudaMalloc((void**)&d_out, d_size*sizeof(float));

    // copy data to device
    // solve problem using device
    // copy result back from device
    // unallocate device arrays
    cudaFree(d_in);
    cudaFree(d_out);
}













/**
 * block_size is the size of the cuda block (must be a multiple 
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU needs to copy it back to host.
 *
 * OP         class denotes the associative binary operator 
 *                and should have an implementation similar to 
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates, 
 *                e.g., float or int. 
 */
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


/**
 * block_size is the size of the cuda block (must be a multiple 
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * flags      is the flag array, in which !=0 indicates 
 *                start of a segment.
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU you need to copy it back to host.
 *
 * OP         class denotes the associative binary operator 
 *                and should have an implementation similar to 
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates, 
 *                e.g., float or int. 
 */
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






int maxSegmentSum(  unsigned int block_size, // block size chosen
                    unsigned int d_size,     // size of calculation
                    int* d_in                // device memory pointer to input array
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


#endif // HOST_LIB
