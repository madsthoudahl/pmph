#ifndef DEV_KERS
#define DEV_KERS

#include <cuda_runtime.h>

#define TILE_SIZE   32    // SMALLER THAN 33
#define BLOCK_SIZE 512    // SMALLER THAN 1025, AND DIVISIBLE BY 32

/*******************************************************************************
 *  ASSIGNMENT 3 KERNELS                                                       *
 ******************************************************************************/

// ASS3 TASK1 -  MATRIX TRANSPOSITION                                         //

template<class T> __global__ void 
transpose_naive_kernel( const unsigned int cols_out, const unsigned int cols_in, T* d_in, T* d_out ) {
    // Implement a “naive” transpose in CUDA, i.e., write a two-dimensional CUDA
    // kernel that exploits both N and M dimensions of parallelism and which 
    // performs the transposition much in the way shown in the pseudo-code

    // NOTE THAT rows_out = cols_in AND cols_out = rows_in;

    const unsigned int xid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int yid = blockIdx.y * blockDim.y + threadIdx.y;
    
    int read_idx  = yid * cols_in + xid;
    int write_idx = xid * cols_out + yid;

    if ( (yid < cols_out) & (xid < cols_in) ) {
        d_out[write_idx] = d_in[read_idx];
    }
}



template<class T> __global__ void 
transpose_opt_kernel( const unsigned int cols_out, const unsigned int cols_in, T* d_in, T* d_out ){
    // NOTE THAT rows_out = cols_in AND cols_out = rows_in;
    // NOTE ALSO blockDim.x = blockDim.y = TILE_SIZE
    __shared__ T tile_mem[TILE_SIZE][TILE_SIZE+1]; // MEMORY BANK FIX ??

    // calculate indexing into global array
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.x + threadIdx.y;

    if ((y < cols_out) & (x < cols_in)) {
        tile_mem[threadIdx.y][threadIdx.x]  = d_in[y * cols_in + x];
    }

    // wait for all threads in block to read before writing
    __syncthreads();
    
    // transpose global indexing
    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;

    if ((y < cols_in) & (x < cols_out)) {
        d_out[y * cols_out + x] = tile_mem[threadIdx.x][threadIdx.y];
    }
    
}







// ASS3 TASK2 -  MATRIX ACCUMULATOR                                           //

template<class T> __global__ void 
mat_acc_first_kernel( const unsigned int rows_in, const unsigned int cols_in, T* d_in, T* d_out )
{
    // Implement quickly a straightforward CUDA version of the program above, 
    // in which the first loop of index i and count N is executed in parallel, 
    // i.e., corresponds to a one-dimensional CUDA kernel, and the second one 
    // is executed sequentially, i.e., it is part of the kernel code
    // COLS = 64  ROWS = N
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < rows_in) {
        T accum =  d_in[gid*cols_in] * d_in[gid*cols_in];
        T tmp;
        d_out[gid*cols_in] = accum;
    
        for (int i=1; i < cols_in; i++) {
            tmp    = d_in[gid * cols_in + i];
            accum  = sqrt(accum) + tmp*tmp;
            d_out[gid * cols_in + i] = accum;
        }
    }
}


template<class T> __global__ void 
mat_acc_second_kernel( const unsigned int rows_in, const unsigned int cols_in, T* d_in, T* d_out )
{
    // Rewrite quickly the CUDA program such that all accesses to global memory
    // are coalesced, i.e., the new program reads from the transpose of A, and 
    // computes the transpose of B:
    // • transpose A in A', using the optimized CUDA implementation of Task I.1.
    // • write a CUDA kernel that implements a modified version of the pseudo-
    //   code above that uses A' instead of A and computes B' (the transpose of B),
    //   instead of B.
    // • finally, after the execution of the CUDA kernel, transpose B' to obtain 
    //   the original result B
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < cols_in) {
        T accum =  d_in[gid]*d_in[gid];
        T tmp;
        d_out[gid] = accum;
        for (int i=1; i < rows_in; i++) {
            tmp    = d_in[gid + cols_in * i];
            accum  = sqrt(accum) + tmp*tmp;
            d_out[gid + cols_in * i] = accum;
        }
    }
}








// ASS3 TASK3 -  MATRIX MULTIPLICATION                                        //

template<class T> __global__ void
matmult_naive_kernel( const unsigned int M,  // outer y limit
                      const unsigned int U,  // k from 0 to u, match dim
                      const unsigned int N,  // outer x limit
                      T*                 A,  // input  MxU
                      T*                 B,  // input  UxN
                      T*                 res // output MxN
) {
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((j < N) & (i < M)) {
        float tmp = 0.0;
            for (int k=0 ; k<U ; k++) { 
                tmp += A[i*U+k] * B[k*N+j];
            }
        res[j + N * i] = tmp;
    }
}



template<class T> __global__ void
matmult_tile_kernel( const unsigned int M,  // outer y limit
                     const unsigned int U,  // k from 0 to u, match dim
                     const unsigned int N,  // outer x limit
                     T*                 A,  // input  MxU
                     T*                 B,  // input  UxN
                     T*                 res // output MxN
) {
    __shared__ T Ash[TILE_SIZE+1][TILE_SIZE+1], Bsh[TILE_SIZE+1][TILE_SIZE+1];
    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int tile = blockDim.x; // = blockDim.y

    const unsigned int jj = blockIdx.x * tile;
    const unsigned int j  = jj + x;
    const unsigned int ii = blockIdx.y * tile;
    const unsigned int i  = ii + y;

    float tmp = 0.0;
    for (int kk=0 ; kk<U ; kk+=tile) {
        Ash[y][x] = ((i<M) && ((kk+x)<U)) ? A[i*U+(kk+x)] : 0.0 ;
        Bsh[y][x] = ((j<N) && ((kk+y)<U)) ? B[(kk+y)*N+j] : 0.0 ;
        __syncthreads();
        for (int k=0 ; k<tile ; k++) {
            tmp += Ash[y][k] * Bsh[k][x];
        }
        __syncthreads();
    }
    if ((j < N) & (i < M)) {
        res[j + N * i] = tmp;
    }
    
}
    



#endif //DEV_KERS



