#ifndef GPU_KSHAPE__
#define GPU_KSHAPE__

#include "cuda_runtime.h"
#include <cstdlib>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "shapeExtract.cuh"
#include "timer.cuh"
#include "check.cuh"
#include "print_test.cuh"

template<typename T>
__device__ void rand_init(unsigned int seed, T &result, const size_t k) 
{
    curandState_t state;
    curand_init(seed, 0, 0, &state);
    result = curand(&state) % k + 1;
}

template<typename T>
__global__ void idx_init(T *d_idx, const size_t k, const size_t mat_row)
{
    const unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < mat_row)
    {
        rand_init(idx, d_idx[idx], k); 
    }
}

template<typename T>
__device__ void get_max_element(T *d_mat_ncc, const size_t ncc_len, T &max_val)
{
    for (size_t i = 0; i < ncc_len; i++)
    {
        max_val = d_mat_ncc[i]>max_val?d_mat_ncc[i]:max_val;
    }
    
}


template<typename T>
__global__ void get_pos_idx(T *d_mat_ncc, int *d_idx,
                            const size_t k, const size_t ncc_len, const size_t mat_row)
{
    const size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < mat_row)
    {
        double max_val = -1e15;
        int pos = 0;
        for (size_t i = 0; i < k; i++)
        {
            T max_tmp = -1e15;
            get_max_element(d_mat_ncc+ncc_len*idx*k+i*ncc_len, ncc_len, max_tmp);
            if(max_val < max_tmp)
            {
                max_val = max_tmp;
                pos = i;
            }
        }
        d_idx[idx] = pos+1;
    }
    
}

template<typename T>
__global__ void is_equal(T *d_old_idx, T *d_idx, const size_t mat_row, T *flag)
{
    const int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < mat_row)
    {
        if(d_old_idx[idx] != d_idx[idx])
             flag[0] = 0;
    }
}



template<typename T>
void k_shape(T *d_mat, int *idx,
             const size_t mat_row, const size_t mat_col, const size_t k,
             int &iter_num)
{
    int *d_idx;
    int *d_old_idx;
    T *d_centers;
    int *d_kcount;
    T *ncc_out;
    const size_t ncc_len = 2*mat_col-1;
    CHECK(cudaMalloc((void**)&d_idx, sizeof(int)*mat_row));
    CHECK(cudaMalloc((void**)&d_old_idx, sizeof(int)*mat_row));
    CHECK(cudaMalloc((void**)&d_centers, sizeof(T)*k*mat_col));
    cudaMemset(d_centers, 0, sizeof(T)*k*mat_col);
    CHECK(cudaMalloc((void**)&d_kcount, sizeof(int)*k));
    CHECK(cudaMalloc((void**)&ncc_out, sizeof(T)*mat_row*k*ncc_len));
    cudaMemset(ncc_out, 0, sizeof(T)*mat_row*k*ncc_len);
    const size_t blockSize = 32;
    const size_t gridSize = (mat_row+blockSize-1)/blockSize;
    
    idx_init<<<gridSize, blockSize>>>(d_idx, k, mat_row);
    cudaDeviceSynchronize();
    
    cudaMemcpy(d_old_idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToDevice);

    int *d_flag;
    CHECK(cudaMalloc((void**)&d_flag, sizeof(int)));
    
    std::cout<<"start iter"<<std::endl;
    GpuTimer timer;
    int flag[1];
    for (int i = 0; i < iter_num; i++)
    {
        std::cout<<"iter ------------------->i: "<<i<<std::endl;

        flag[0] = 1;
        cudaMemcpy(d_flag, flag, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_kcount, 0, sizeof(int)*k);       
        
        std::cout<<"start calcute extract_shape"<<std::endl;
        timer.Start();
        extract_shape(d_idx, d_mat, d_kcount, d_centers, k, mat_row, mat_row, mat_col);
        timer.Stop();
        
        printf("extract_shape run on GPU: %f msecs.\n", timer.Elapsed());
        //err in centers
        //print_test_f<<<1,1>>>(d_centers, k, mat_col);
        
        std::cout<<"start calcute NCC_3D"<<std::endl;
        timer.Start();
        NCC_3D(d_mat, d_centers, ncc_out, mat_row, k, mat_col);
        timer.Stop();
        printf("NCC_3D run on GPU: %f msecs.\n", timer.Elapsed());
         
        std::cout<<"start calcute get_pos_idx"<<std::endl;
        timer.Start();
        get_pos_idx<<<gridSize, blockSize>>>(ncc_out, d_idx, k, ncc_len, mat_row);
        cudaDeviceSynchronize(); 
        timer.Stop();
        printf("get_pos_idx run on GPU: %f msecs.\n", timer.Elapsed());  

        std::cout<<"start calcute is_equal"<<std::endl;
        timer.Start();
        is_equal<<<gridSize, blockSize>>>(d_old_idx, d_idx, mat_row, d_flag);
        cudaDeviceSynchronize();
        cudaMemcpy(flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        timer.Stop();
        printf("is_equal run on GPU: %f msecs.\n", timer.Elapsed());

        if (flag[0])
        {
            iter_num = i+1;
            break;
        }  
        else
            cudaMemcpy(d_old_idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToDevice);
        std::cout<<std::endl;
    }

    std::cout<<"iter end"<<std::endl;
    
    cudaMemcpy(idx, d_idx, sizeof(int)*mat_row, cudaMemcpyDeviceToHost);

    cudaFree(d_flag);
    cudaFree(d_old_idx);
    cudaFree(ncc_out);
    cudaFree(d_idx);
    cudaFree(d_centers);
    cudaFree(d_kcount);
}

#endif