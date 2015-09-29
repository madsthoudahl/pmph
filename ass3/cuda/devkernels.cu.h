#ifndef DEV_KERS
#define DEV_KERS

#include <cuda_runtime.h>


/*******************************************************************************
 *  ASSIGNMENT 3 KERNELS                                                       *
 ******************************************************************************/

// ASS3 TASK1 -  MATRIX TRANSPOSITION                                         //
template<class T> __global__ void 
transpose_naive_kernel( const unsigned int rows_in, const unsigned int cols_in, T* d_in, T* d_out ){
    // TODO fix implementation
    // Implement a “naive” transpose in CUDA, i.e., write a two-dimensional CUDA
    // kernel that exploits both N and M dimensions of parallelism and which 
    // performs the transposition much in the way shown in the pseudo-code
    const unsigned int xid = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int yid = blockIdx.y*blockDim.y + threadIdx.y;
    // map the two 2D indices to a single linear, 1D index
    int grid_width  = gridDim.x * blockDim.x;
    int grid_height = gridDim.y * blockDim.y;
    int l_idx = yid * grid_width  + xid;
    int s_idx = xid * grid_height + yid;

    if ((yid < cols_in) & (xid < rows_in)) {
        d_out[s_idx] = d_in[l_idx];
    }
}

template<class T> __global__ void 
transpose_opt_kernel( const unsigned int rows_in, const unsigned int cols_in, T* d_in, T* d_out ){
    // TODO fix implementation
    const unsigned int xid = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int yid = blockIdx.y*blockDim.y + threadIdx.y;
    if ((yid < cols_in) & (xid < rows_in)) {
        d_out[yid*rows_in+xid] = d_in[xid*cols_in+yid] * 2;
    }
}

// ASS3 TASK2 -  MATRIX MULTIPLICATION                                        //

// ASS3 TASK3 -  MATRIX MULTIPLICATION                                        //






template<class T>
class Add {
  public:
    typedef T BaseType;
    static __device__ __host__ inline T identity()                    { return (T)0;    }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }
};

template<class T>
class Mul {
  public:
    typedef T BaseType;
    static __device__ inline T identity()                      { return (T)1;    }
    static __device__ inline T apply(const T& t1, const T& t2) { return t1 * t2; }
};

class MyInt4 {
  public:
    int x; int y; int z; int w;

    __device__ __host__ inline MyInt4() {
        x = 0; y = 0; z = 0; w = 0; 
    }

    __device__ __host__ inline MyInt4(const int& a, const int& b, const int& c, const int& d) {
        x = a; y = b; z = c; w = d; 
    }

    __device__ __host__ inline MyInt4(const MyInt4& i4) { 
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
    }

    volatile __device__ __host__ inline MyInt4& operator=(const MyInt4& i4) volatile {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
        return *this;
    }
};

class MsspOp {
  public:
    typedef MyInt4 BaseType;
    static __device__ inline MyInt4 identity() { return MyInt4(0,0,0,0); }  
    static __device__ inline MyInt4 apply(volatile MyInt4& t1, volatile MyInt4& t2) { 
        int mss = max( max( t1.x, t2.x ),( t1.z + t2.y )); // max segment sum
        int mis = max( t1.y, ( t1.w + t2.y ));             // max initial sum
        int mcs = max(( t1.z + t2.w ), t2.z);              // max conclusive sum
        int t   = (t1.w+ t2.w);                            // total sum
        return MyInt4(mss, mis, mcs, t); 
    }
};


/*******************************************/
/*** MaxSegmentSum Helper & Kernel       ***/
/*******************************************/
__global__ void 
createMyInt4Kernel(int* d_in, MyInt4* d_out, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (gid < d_size)  d_out[gid] = MyInt4( d_in[gid], d_in[gid], d_in[gid], d_in[gid]);
}


__global__ void 
extractLastKernel(MyInt4* d_in, int* d_out, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (gid == (d_size-1)) d_out[0] = d_in[gid].x;  
}


/***************************************/
/*** Scan Inclusive Helpers & Kernel ***/
/***************************************/
template<class OP, class T>
__device__ inline
T scanIncWarp( volatile T* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  ptr[idx] = OP::apply(ptr[idx-1],  ptr[idx]); 
    if (lane >= 2)  ptr[idx] = OP::apply(ptr[idx-2],  ptr[idx]);
    if (lane >= 4)  ptr[idx] = OP::apply(ptr[idx-4],  ptr[idx]);
    if (lane >= 8)  ptr[idx] = OP::apply(ptr[idx-8],  ptr[idx]);
    if (lane >= 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);

    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T>
__device__ inline
T scanIncBlock(volatile T* ptr, const unsigned int idx) {
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;

    T val = scanIncWarp<OP,T>(ptr,idx);
    __syncthreads();

    // place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == 31) { ptr[warpid] = const_cast<T&>(ptr[idx]); } 
    __syncthreads();

    //
    if (warpid == 0) scanIncWarp<OP,T>(ptr, idx);
    __syncthreads();

    if (warpid > 0) {
        val = OP::apply(ptr[warpid-1], val);
    }

    return val;
}

template<class OP, class T>
__global__ void 
scanIncKernel(T* d_in, T* d_out, unsigned int d_size) {
    extern __shared__ char sh_mem1[];
    volatile T* sh_memT = (volatile T*)sh_mem1;
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;
    T el    = (gid < d_size) ? d_in[gid] : OP::identity();
    sh_memT[tid] = el;
    __syncthreads();
    T res   = scanIncBlock < OP, T >(sh_memT, tid);
    if (gid < d_size) d_out [gid] = res; 
}


/***********************************************************/
/*** Kernels to copy/distribute the end of block results ***/
/***********************************************************/

template<class T>
__global__ void 
copyEndOfBlockKernel(T* d_in, T* d_out, unsigned int d_out_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < d_out_size)
        d_out[gid] = d_in[ blockDim.x*(gid+1) - 1];
}

template<class OP, class T>
__global__ void 
distributeEndBlock(T* d_in, T* d_out, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < d_size && blockIdx.x > 0)
        d_out[gid] = OP::apply(d_out[gid],d_in[blockIdx.x-1]);
}

template<class T>
__global__ void 
shiftRightByOne(T* d_in, T* d_out, T ne, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if      (gid == 0)      d_out[gid] = ne;
    else if (gid < d_size)  d_out[gid] = d_in[gid-1];
}

/*************************************************/
/*************************************************/
/*** Segmented Inclusive Scan Helpers & Kernel ***/
/*************************************************/
/*************************************************/
template<class OP, class T, class F>
__device__ inline
T sgmScanIncWarp(volatile T* ptr, volatile F* flg, const unsigned int idx) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-1], ptr[idx]); }
        flg[idx] = flg[idx-1] | flg[idx];
    }
    if (lane >= 2)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-2], ptr[idx]); }
        flg[idx] = flg[idx-2] | flg[idx];
    }
    if (lane >= 4)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-4], ptr[idx]); }
        flg[idx] = flg[idx-4] | flg[idx];
    }
    if (lane >= 8)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-8], ptr[idx]); }
        flg[idx] = flg[idx-8] | flg[idx];
    }
    if (lane >= 16)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]); }
        flg[idx] = flg[idx-16] | flg[idx];
    }

    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T, class F>
__device__ inline
T sgmScanIncBlock(volatile T* ptr, volatile F* flg, const unsigned int idx) {
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;
    const unsigned int warplst= (warpid<<5) + 31;

    // 1a: record whether this warp begins with an ``open'' segment.
    bool warp_is_open = (flg[(warpid << 5)] == 0);
    __syncthreads();

    // 1b: intra-warp segmented scan for each warp
    T val = sgmScanIncWarp<OP,T>(ptr,flg,idx);

    // 2a: the last value is the correct partial result
    T warp_total = const_cast<T&>(ptr[warplst]);
    
    // 2b: warp_flag is the OR-reduction of the flags 
    //     in a warp, and is computed indirectly from
    //     the mindex in hd[]
    bool warp_flag = flg[warplst]!=0 || !warp_is_open;
    bool will_accum= warp_is_open && (flg[idx] == 0);

    __syncthreads();

    // 2c: the last thread in a warp writes partial results
    //     in the first warp. Note that all fit in the first
    //     warp because warp = 32 and max block size is 32^2
    if (lane == 31) {
        ptr[warpid] = warp_total; //ptr[idx]; 
        flg[warpid] = warp_flag;
    }
    __syncthreads();

    // 
    if (warpid == 0) sgmScanIncWarp<OP,T>(ptr, flg, idx);
    __syncthreads();

    if (warpid > 0 && will_accum) {
        val = OP::apply(ptr[warpid-1], val);
    }
    return val;
}

template<class OP, class T>
__global__ void 
sgmScanIncKernel(T* d_in, int* flags, T* d_out, 
                          int* f_rec, T* d_rec, unsigned int d_size) {
    extern __shared__ char sh_mem[];
    volatile T*   vals_sh = (volatile T*)sh_mem;
    volatile int* flag_sh = (int*) (vals_sh + blockDim.x);
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;
    int fl;   
    if (gid < d_size) { vals_sh[tid] = d_in[gid];      fl = flags[gid]; }
    else              { vals_sh[tid] = OP::identity(); fl = 0;          }
    flag_sh[tid] = fl;
    __syncthreads();
    T res = sgmScanIncBlock <OP, T>(vals_sh, flag_sh, tid);
    if (gid < d_size) d_out [gid] = res; 

    // set the flags and data for the recursive step!
    if(tid == 0)  { f_rec[blockIdx.x] = 0; }
    __syncthreads();
    if(fl  >  0)  { f_rec[blockIdx.x] = 1; }
    if(tid == (blockDim.x - 1)) { d_rec[blockIdx.x] = res; }
}

template<class OP, class T>
__global__ void 
sgmDistributeEndBlock(T* d_rec_in, T* d_out, int* f_inds, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < d_size && blockIdx.x > 0) {
        if(f_inds[gid] == 0)
            d_out[gid] = OP::apply(d_out[gid], d_rec_in[blockIdx.x-1]);
    }
}

////////////////////////////////////////
////////////////////////////////////////

/**
 * d_in     the input array
 * flags    the flag array
 * d_out    the result array: if the corresponding flag is set then ne
 *                            else the previous element of d_in
 * ne       is the neutral element
 * d_size   if the size of the input, flag, and output arrays
 **/
template<class T>
__global__ void 
sgmShiftRightByOne(T* d_in, int*flags, T* d_out, T ne, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if      (gid == 0)        d_out[gid] = ne;
    else if (gid < d_size)    d_out[gid] = d_in[gid-1];
    if (flags[gid] != 0) d_out[gid] = ne;
}


/**
 * This implements the map from MSSP:
 * inp_d    the original array (of ints)
 * inp_lift the result array, in which an integer x in inp_d
 *              should be transformed to MyInt4(x,x,x,x) if x > 0
 *                                and to MyInt4(0,0,0,x) otherwise 
 * inp_size is the size of the original (and output) array
 *              in number of int (MyInt4) elements
 **/
__global__ void 
msspTrivialMap(int* inp_d, MyInt4* inp_lift, int inp_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < inp_size) {
        int x = inp_d[gid];
        if ( x > 0 ) inp_lift[gid] = MyInt4 ( x, x, x, x);
        else inp_lift[gid] = MyInt4 ( 0, 0, 0, x);
    }
}

/**
 * mat_inds  the column indices corresponding to the values in `mat_vals'
 * mat_vals  the values of the matrix
 * vct       the values of the vector
 * tot_size  the total number of (non-zero) elements of the matrix
 * tmp_pairs the result array: should hold the mutiplication between 
 *              each matrix (non-zero) value and the corresponding vct element,
 *              which is found via mat_inds.
 */
__global__ void 
spMatVctMult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < tot_size) {
        tmp_pairs[gid] = vct[ mat_inds[gid] ] * mat_vals[gid];
    }
}

/**
 * tmp_scan  the segmented scan (+) version of the flat matrix,
 *               i.e., the last element of each segment is the
 *               result vector value. 
 * tmp_inds  IF `gid' is the index of the LAST ELEMENT of a SEGMENT (row) 
 *               then (tmp_inds[gid]-1) is the index in `vct_res' where that
 *               element should be stored, i.e., 
 *               vct_res[tmp_inds[gid]-1] = tmp_scan[gid]
 * flags_d   denote the starts of the rows of the flat matrix, i.e., 
 *               flags[gid]!=0 then a new row starts at position `gid'.
 *               It follows that an index `gid' corresponds to the 
 *               last element of a row if flags_d[gid+1] != 0
 * tot_size  the total number of elements of the flat matrix
 * vct_res   the result vector
 */
__global__ void
write_lastSgmElem(float* tmp_scan, int* tmp_inds, int* flags_d, int tot_size, float* vct_res) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < tot_size) {
        if ( flags_d[gid+1] != 0 )     // all but the last dotproduct
            vct_res[tmp_inds[gid]-1] = tmp_scan[gid];
    } else if (gid == (tot_size-1)) {  // the last dotproduct
               vct_res[tmp_inds[gid]-1] = tmp_scan[gid];
    }
}


#endif //DEV_KERS

