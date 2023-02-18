#ifndef GPU_CENTER__
#define GPU_CENTER__

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusolverDn.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "SBD.cuh"
#include "normalization.cuh"
#include "timer.cuh"
#include "check.cuh"
#include "print_test.cuh"
#include "readfile.cuh"

#define TILE_WIDTH 16

//一行为一个线程
template<typename T>
__global__ void class_count(T *d_idx, const size_t idx_len, T *k_count)
{
    const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < idx_len)
    {
        atomicAdd(&k_count[d_idx[idx] - 1], 1);
    }
}

template<typename T>
__global__ void histo_split(T *d_idx, T *count_tmp,
                           const size_t step_size, const size_t k, const size_t length)
{
    const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    const size_t stride = (length%step_size == 0) ? (length/step_size) : (length/step_size+1);
    size_t step = step_size;
    if(idx*step_size < length && (idx+1)*step_size > length)
        step = length%step_size;
    if(idx*step_size < length)
    {
        T *cur_count = (T*)malloc(sizeof(T)*k);
        memset(cur_count, 0, sizeof(T)*k);
        for (size_t i = 0; i < step; i++)
        {
            cur_count[d_idx[idx*step_size+i] - 1]++;
        }
        for (size_t i = 0; i < k; i++)
        {
            count_tmp[stride*i + idx] = cur_count[i];
            
        }
        
        free(cur_count);
    }
}


template<typename T>
__device__ void sum_reduce(T *arr, const size_t arr_size, T &sum)
{
    for (size_t i = 0; i < arr_size; i++)
    {
        sum += arr[i];
    }
    
}

template<typename T>
__global__ void histo_merge(T *d_idx, T *count_tmp, T *d_kcount,
                           const size_t step_size, const size_t k, const size_t length)
{
    const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    const size_t stride = (length%step_size == 0) ? (length/step_size) : (length/step_size+1);
    if(idx < k)
    {
        sum_reduce(count_tmp+idx*stride, stride, d_kcount[idx]);
    }
}

// two levels hash
template<typename T>
__global__ void get_loc2(T *d_idx, T *count_k_cur, T *loc2, const size_t idx_len)
{
    for (size_t i = 0; i < idx_len; i++)
    {
        loc2[i] = count_k_cur[d_idx[i] - 1];
        count_k_cur[d_idx[i] - 1]++;
    } 
}

template<typename T>
__global__ void loc2_split(T *d_idx, T *d_count_tmp, T *d_loc2,
                           const size_t step_size, const size_t k, const size_t length)
{
    const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    const size_t stride = (length%step_size == 0) ? (length/step_size) : (length/step_size+1);
    size_t step = step_size;
    if(idx*step_size < length && (idx+1)*step_size > length)
        step = length%step_size;
    if(idx*step_size < length)
    {
        T *cur_count = (T*)malloc(sizeof(T)*k);
        memset(cur_count, 0, sizeof(T)*k);
        for (size_t i = 0; i < step; i++)
        {
            d_loc2[idx*step_size+i] = cur_count[d_idx[idx*step_size+i] - 1];
            cur_count[d_idx[idx*step_size+i] - 1]++;
        }

        //map cur_count to count_tmp
        for (size_t i = 0; i < k; i++)
        {
            d_count_tmp[i*stride+idx] = cur_count[i];
        }
        free(cur_count);
    }
    
}

template<typename T>
__device__ void exclusive_scan(T *arr, const size_t arr_size)
{
    T tmp_cur, tmp_next;
    tmp_cur = arr[0];
    arr[0] = 0;
    for (size_t i = 1; i < arr_size; i++)
    {
        tmp_next = arr[i];
        arr[i] = tmp_cur + arr[i - 1];
        tmp_cur = tmp_next;
    }
}

template<typename T>
__global__ void per_scan(T *d_count_tmp, const size_t step_size, const size_t k, const size_t length)
{
    const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    const size_t stride = (length%step_size == 0) ? (length/step_size) : (length/step_size+1);
    if(idx < k)
    {
        //thrust::exclusive_scan(thrust::device, d_count_tmp+idx*stride, 
        //                       d_count_tmp+(idx+1)*stride, d_count_tmp+idx*stride);
        exclusive_scan(d_count_tmp+idx*stride, stride);
    }
}

template<typename T>
__global__ void loc2_merge(T *d_idx, T *d_count_tmp, T *d_loc2,
                           const size_t step_size, const size_t k, const size_t length)
{
    const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    const size_t stride = (length%step_size == 0) ? (length/step_size) : (length/step_size+1);
    size_t step = step_size;
    if(idx*step_size < length && (idx+1)*step_size > length)
        step = length%step_size;
    if(idx*step_size < length)
    {
        T *cur_count = (T*)malloc(sizeof(T)*k);
        memset(cur_count, 0, sizeof(T)*k);
        //map count_tmp to cur_count
        for (size_t i = 0; i < k; i++)
        {
            cur_count[i] = d_count_tmp[i*stride+idx];
        }

        for (size_t i = 0; i < step; i++)
        {
            d_loc2[idx*step_size+i] += cur_count[d_idx[idx*step_size+i] - 1];
        }
        free(cur_count);
    }
}

// hash sort --- using 2D blocks and threads
template<typename T>
__global__ void block_alloc2D(T *d_mat_seg, T *d_mat, int *d_idx, int *d_scan_kcount,
                              int *d_loc2, const size_t mat_row, const size_t mat_col)
{
    const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    const size_t idy = threadIdx.y + blockDim.y*blockIdx.y;
    if (idx < mat_row && idy < mat_col)
    {
        const size_t pos0 = d_idx[idx] - 1;
        const size_t pos1 = d_scan_kcount[pos0]*mat_col + mat_col*d_loc2[idx];
        d_mat_seg[pos1 + idy] = d_mat[idx*mat_col+idy];
    }
    
}

//copy first sample to check eigenvector pos or neg
template<typename T>
__global__ void val_copy(T *d_mat_seg, T *d_mat_val, const size_t mat_col)
{
    //const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    memcpy(d_mat_val, d_mat_seg, sizeof(T)*mat_col);
}

//matrix Q 
template<typename T>
__global__ void get_matQ(T *d_matQ, const size_t mat_row, const size_t mat_col)
{
    const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    const size_t idy = threadIdx.y + blockDim.y*blockIdx.y;
    if(idx < mat_row && idy < mat_col)
    {
        T a = (-1)*(1.0/mat_col);
        T b = 1 - 1.0/mat_col;
        d_matQ[idx + idy*mat_col] = idx == idy ? b:a;
    }
}

template<typename T>
__device__ void ops_center(T *d_centers, const size_t MAT_COLS)
{
    for (size_t i = 0; i < MAT_COLS; i++)
    {
         d_centers[i] = -1 * d_centers[i];
    }
}

//judge eigenvector pos or neg
template<typename T>
__device__ void judge_center(T *d_vals, T *d_centers, const size_t MAT_COLS, 
                             bool flag, T &finddistance)
{
    if(flag)
    {
        for (size_t idx = 0; idx < MAT_COLS; idx++)
        {
            finddistance += (d_vals[idx] - d_centers[idx])*(d_vals[idx] - d_centers[idx]); 
        }
        finddistance = sqrtf(finddistance);    
    }
    else
    {
        for (size_t idx = 0; idx < MAT_COLS; idx++)
        {
            finddistance += (d_vals[idx] + d_centers[idx])*(d_vals[idx] + d_centers[idx]);
        }
        finddistance = sqrtf(finddistance);     
    }
     
}


template<typename T>
__global__ void copy_mat_ser(int *d_idx, int id_cur, int class_iter, 
                            int *seg_cur, size_t mat_col, T *d_mat_seg, T *d_mat)
{
    if(d_idx[id_cur] == class_iter)
    {
        memcpy(d_mat_seg+seg_cur[0]*mat_col, d_mat+id_cur*mat_col, sizeof(T)*mat_col);
        seg_cur[0] += 1;
    }
}


template<typename T>
__global__ void copy_mat_ser_mut(int *d_idx, int *d_seg_cur, size_t mat_row, int class_iter,
                                 size_t mat_col, T *d_mat_seg, T *d_mat)
{
    for(int i=0; i<mat_row; i++)
    {
        if(d_idx[i] == class_iter)
        {
            memcpy(d_mat_seg+d_seg_cur[0]*mat_col, d_mat+i*mat_col, sizeof(T)*mat_col);
            d_seg_cur[0] += 1;
        }
    }
    
}


template<typename T>
__global__ void get_center(T *d_vals, T *d_EigVecs, T *d_centers, 
                            int class_cur, const size_t MAT_COLS)
{
        T finddistance1 = 0;
        T finddistance2 = 0;
        judge_center(d_vals, d_EigVecs+(MAT_COLS-1)*MAT_COLS, 
                     MAT_COLS, true, finddistance1);
        judge_center(d_vals, d_EigVecs+(MAT_COLS-1)*MAT_COLS, 
                     MAT_COLS, false, finddistance2);
        if(finddistance1 >= finddistance2)
        {
            ops_center(d_EigVecs+(MAT_COLS-1)*MAT_COLS, MAT_COLS);
        }
        
        memcpy(d_centers+(class_cur-1)*MAT_COLS, 
               d_EigVecs+(MAT_COLS-1)*MAT_COLS, 
               sizeof(T)*MAT_COLS);
}

//calcute eigenvector to get centrious
template <typename T>
typename std::enable_if<(sizeof(T) == 4)>::type get_class_centers(float *d_mat_seg, float *d_vals, 
                                                                  float *d_centers,
                                                                  int k_count_cur, const size_t k, 
                                                                  const size_t MAT_COLS, int class_cur)
{
    
    float *d_result;
    CHECK(cudaMalloc((void**)&d_result, sizeof(float)*MAT_COLS*MAT_COLS)); 
    cudaMemset(d_result, 0, sizeof(float)*MAT_COLS*MAT_COLS);

    cublasHandle_t handle;
    cusolverDnHandle_t cusolverH;

    cublasStatus_t blasStat;
    cusolverStatus_t solverStat;

    
    blasStat = cublasCreate(&handle);
    if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("shapeExtract CUBLAS initialization failed%d\n",blasStat);exit( -1 );}
    solverStat = cusolverDnCreate(&cusolverH);
    if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("shapeExtract CUSOLVER initialization failed%d\n",solverStat);exit( -1 );}
    
        
    float alpha = 1.0, beta = 0.0;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((MAT_COLS+threadsPerBlock.x-1)/threadsPerBlock.x, (MAT_COLS+threadsPerBlock.y-1)/threadsPerBlock.y);   


    if(k_count_cur != 0)
    {
        
        blasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,   
                               MAT_COLS, MAT_COLS, k_count_cur,          
                               &alpha, d_mat_seg, MAT_COLS,          
                               d_mat_seg, MAT_COLS, &beta,             
                               d_result, MAT_COLS);
        if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}
        
        
        T *d_matQ;
        CHECK(cudaMalloc((void**)&d_matQ, sizeof(T)*MAT_COLS*MAT_COLS));
        get_matQ<<<numBlocks, threadsPerBlock>>>(d_matQ, MAT_COLS, MAT_COLS);
        cudaDeviceSynchronize();
        //Q*S
        blasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   
                               MAT_COLS, MAT_COLS, MAT_COLS,          
                               &alpha, d_result, MAT_COLS,          
                               d_matQ, MAT_COLS, &beta,             
                               d_result, MAT_COLS);
        if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}
        //(Q*S)*Q
        blasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,   
                    MAT_COLS, MAT_COLS, MAT_COLS,          
                    &alpha, d_matQ, MAT_COLS,          
                    d_result, MAT_COLS, &beta,             
                    d_result, MAT_COLS);
        if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}
        cudaFree(d_matQ);
        float *d_EigVal; 
        float *d_work;
        int *devInfo; 
	    int lwork = 0;
        CHECK(cudaMalloc((void**)&d_EigVal, sizeof(float)*MAT_COLS));
        CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
	    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	    solverStat = cusolverDnSsyevd_bufferSize(cusolverH, jobz, uplo, MAT_COLS, 
                                                 d_result, MAT_COLS, 
                                                 d_EigVal, &lwork);
        if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER set buffer failed%d\n",solverStat);exit( -1 );}
        
	    CHECK(cudaMalloc((void**)&d_work, sizeof(float)*lwork));
        solverStat = cusolverDnSsyevd(cusolverH, jobz, uplo, MAT_COLS, 
                                      d_result, MAT_COLS, 
                                      d_EigVal, d_work, lwork, devInfo);
        if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER cusolverDnSsyevd failed%d\n",solverStat);exit( -1 );}
        
        cudaFree(d_EigVal);
        cudaFree(d_work);
        cudaFree(devInfo);
    }
    cublasDestroy(handle);
    cusolverDnDestroy(cusolverH);
    
    

    get_center<<<1, 1>>>(d_vals, d_result, d_centers, 
                                        class_cur, MAT_COLS);
    cudaDeviceSynchronize();
    cudaFree(d_result);
}






template<typename T>
void extract_shape(int *d_idx, T *d_mat, int *d_k_count, T *d_centers,
                   const size_t k, const size_t idx_len, const size_t mat_row,
                   const size_t mat_col)
{
    //统计每个类别数目
    class_count<int><<<(idx_len + 256 -1)/256, 256>>>(d_idx, idx_len, d_k_count);
    cudaDeviceSynchronize();
    
    int *k_count = (int*)malloc(sizeof(int)*k); 
    cudaMemcpy(k_count, d_k_count, sizeof(int)*k, cudaMemcpyDeviceToHost);

    for (int i = 1; i <= k; i++)
    {
        std::cout<<"k: "<<i<<endl;
        if(k_count[i-1] != 0)
        {
            int *d_seg_cur;
            CHECK(cudaMalloc((void**)&d_seg_cur, sizeof(int)));
            cudaMemset(d_seg_cur, 0, sizeof(int)); 

            T *d_mat_seg;
            CHECK(cudaMalloc((void**)&d_mat_seg, sizeof(T)*k_count[i-1]*mat_col));
            //std::cout<<"k is: "<<k<<endl; 
        
            copy_mat_ser_mut<<<1,1>>>(d_idx, d_seg_cur, mat_row, i, mat_col, d_mat_seg, d_mat);
            cudaDeviceSynchronize();

            /*
            for (int j = 0; j < mat_row; j++)
            {
                copy_mat_ser<<<1,1>>>(d_idx, j, i, d_seg_cur, mat_col, d_mat_seg, d_mat);
            }
            */

            T center_sum = thrust::reduce(thrust::device, d_centers+mat_col*(i-1), d_centers+mat_col*i, 0);
            cudaDeviceSynchronize();
            center_sum = center_sum>0 ? center_sum : (-1)*center_sum;
            size_t cur_count = k_count[i-1];
            if(center_sum > 1e-16)
            {
                sbd3D(d_mat_seg, d_centers+mat_col*(i-1), d_mat_seg, cur_count, mat_col);
            }


            T *d_mat_val;
            CHECK(cudaMalloc((void**)&d_mat_val, sizeof(T)*mat_col));
            val_copy<<<1, 1>>>(d_mat_seg, d_mat_val, mat_col);
            cudaDeviceSynchronize();

            z_norm_gpu_x(d_mat_seg, k_count[i-1], mat_col, 1);

            get_class_centers<T>(d_mat_seg, d_mat_val, d_centers, 
                        k_count[i-1], k, mat_col, i);
        
            cudaFree(d_mat_val);
            cudaFree(d_mat_seg);
            cudaFree(d_seg_cur);
        }
        
    }
    free(k_count);
    
}


#endif