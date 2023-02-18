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
__global__ void val_copy(T *d_mat_seg, T *d_mat_val, int *d_scan_kcount,
                         const size_t k, const size_t mat_col)
{
    const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < k)
    {
        memcpy(d_mat_val+idx*mat_col, d_mat_seg+d_scan_kcount[idx]*mat_col, sizeof(T)*mat_col);
    }
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
__global__ void get_center(T *d_vals, T *d_EigVecs, T *d_centers, int *d_scan_kcount,
                          const size_t k, const size_t MAT_COLS)
{
    const size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < k)
    {

        T finddistance1 = 0;
        T finddistance2 = 0;
        judge_center(d_vals+idx*MAT_COLS, d_EigVecs+idx*MAT_COLS*MAT_COLS+(MAT_COLS-1)*MAT_COLS, 
                     MAT_COLS, true, finddistance1);
        judge_center(d_vals+idx*MAT_COLS, d_EigVecs+idx*MAT_COLS*MAT_COLS+(MAT_COLS-1)*MAT_COLS, 
                     MAT_COLS, false, finddistance2);
        if(finddistance1 >= finddistance2)
        {
            ops_center(d_EigVecs+idx*MAT_COLS*MAT_COLS+(MAT_COLS-1)*MAT_COLS, MAT_COLS);
        }
        
        memcpy(d_centers+idx*MAT_COLS, 
               d_EigVecs+idx*MAT_COLS*MAT_COLS+(MAT_COLS-1)*MAT_COLS, 
               sizeof(T)*MAT_COLS);
    }
}


/*
template<typename T>
__global__ void ED_centers(T *d_mat_seg, T *d_centers, const int classNum, 
                           const int preScanNum, const size_t MAT_COLS)
{
    const size_t idx = threadIdx.x+blockIdx.x*blockDim.x;
    const size_t idy = threadIdx.y+blockDim.y*blockIdx.y;
    if(idy < classNum && idx < MAT_COLS)
    {
        for (int strize = 1; strize < classNum; strize <<= 1)
	    {
		    int idx = idy*strize * 2;
 
		    if (idx < blockDim.x)
			    data[idx]+= data[idx+strize];
 
		    __syncthreads();
	    }
        
    }
}
*/


//calcute eigenvector to get centrious
template <typename T>
typename std::enable_if<(sizeof(T) == 4)>::type get_class_centers(float *d_mat_seg, float *d_vals, 
                                                                  float *d_centers,
                                                                  int *d_scan_kcount, int *scan_kcount, 
                                                                  int *k_count, const size_t k, 
                                                                  const size_t MAT_COLS)
{
    
    float *d_result;
    CHECK(cudaMalloc((void**)&d_result, sizeof(float)*k*MAT_COLS*MAT_COLS)); 
    cudaMemset(d_result, 0, sizeof(float)*k*MAT_COLS*MAT_COLS);

    cudaStream_t *streams = (cudaStream_t*)malloc(k*sizeof(cudaStream_t));
    cublasHandle_t *handle = (cublasHandle_t*)malloc(k*sizeof(cublasHandle_t));
    cusolverDnHandle_t *cusolverH = (cusolverDnHandle_t*)malloc(k*sizeof(cusolverDnHandle_t));

    cublasStatus_t blasStat;
    cudaError_t streamStat;
    cusolverStatus_t solverStat;

    for(size_t i=0; i<k; i++)
    {
         blasStat = cublasCreate(&handle[i]);
         if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("shapeExtract CUBLAS initialization failed%d\n",blasStat);exit( -1 );}
         streamStat = cudaStreamCreate(&streams[i]);
         if (streamStat != cudaSuccess) {printf ("shapeExtract cudaStream initialization failed%d\n",solverStat);exit( -1 );}
         solverStat = cusolverDnCreate(&cusolverH[i]);
         if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("shapeExtract CUSOLVER initialization failed%d\n",solverStat);exit( -1 );}
    }
        
    float alpha = 1.0, beta = 0.0;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((MAT_COLS+threadsPerBlock.x-1)/threadsPerBlock.x, (MAT_COLS+threadsPerBlock.y-1)/threadsPerBlock.y);   


    for(size_t i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            blasStat = cublasSetStream(handle[i], streams[i]);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS stream set failed%d\n",blasStat);exit( -1 );}

            blasStat = cublasSgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_T,   
                                   MAT_COLS, MAT_COLS, k_count[i],          
                                   &alpha, d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS,          
                                   d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS, &beta,             
                                   d_result+i*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}
            
            
            //err at here
            /*
            print_test_f<<<1,1>>>(d_result+i*MAT_COLS*MAT_COLS, 1,i+1);
            if(i == 0)
            {
                //print_test_f<<<1,1>>>(d_mat_seg+MAT_COLS*scan_kcount[i], k_count[i],MAT_COLS);
                //print_test_sumf<<<1,1>>>(d_mat_seg+MAT_COLS*scan_kcount[i], k_count[i],MAT_COLS);
            }
            */

            T *d_matQ;
            CHECK(cudaMalloc((void**)&d_matQ, sizeof(T)*MAT_COLS*MAT_COLS));
            get_matQ<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(d_matQ, MAT_COLS, MAT_COLS);
            cudaDeviceSynchronize();

            //Q*S
            blasStat = cublasSgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N,   
                                   MAT_COLS, MAT_COLS, MAT_COLS,          
                                   &alpha, d_result+i*MAT_COLS*MAT_COLS, MAT_COLS,          
                                   d_matQ, MAT_COLS, &beta,             
                                   d_result+i*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}

            //(Q*S)*Q
            blasStat = cublasSgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N,   
                        MAT_COLS, MAT_COLS, MAT_COLS,          
                        &alpha, d_matQ, MAT_COLS,          
                        d_result+i*MAT_COLS*MAT_COLS, MAT_COLS, &beta,             
                        d_result+i*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasSgemm failed%d\n",blasStat);exit( -1 );}

            cudaFree(d_matQ);

            //print_test_f<<<1,1>>>(d_result+i*MAT_COLS*MAT_COLS, MAT_COLS,MAT_COLS);

            solverStat = cusolverDnSetStream(cusolverH[i], streams[i]);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER set stream failed%d\n",solverStat);exit( -1 );}
            float *d_EigVal; 
            float *d_work;
            int *devInfo; 
	        int lwork = 0;

            CHECK(cudaMalloc((void**)&d_EigVal, sizeof(float)*MAT_COLS));
            CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
	        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	        solverStat = cusolverDnSsyevd_bufferSize(cusolverH[i], jobz, uplo, MAT_COLS, 
                                                     d_result+i*MAT_COLS*MAT_COLS, MAT_COLS, 
                                                     d_EigVal, &lwork);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER set buffer failed%d\n",solverStat);exit( -1 );}
            
	        CHECK(cudaMalloc((void**)&d_work, sizeof(float)*lwork));

            solverStat = cusolverDnSsyevd(cusolverH[i], jobz, uplo, MAT_COLS, 
                                          d_result+i*MAT_COLS*MAT_COLS, MAT_COLS, 
                                          d_EigVal, d_work, lwork, devInfo);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER cusolverDnSsyevd failed%d\n",solverStat);exit( -1 );}
            
            cudaFree(d_EigVal);
            cudaFree(d_work);
            cudaFree(devInfo);
        }

        cublasDestroy(handle[i]);
        cusolverDnDestroy(cusolverH[i]);
        cudaStreamDestroy(streams[i]);
        
    }

    const size_t blockSize = 32;
    const size_t gridSize  = (k+blockSize-1)/blockSize;
    get_center<<<gridSize, blockSize>>>(d_vals, d_result, d_centers, 
                                        d_scan_kcount, k, MAT_COLS);
    cudaDeviceSynchronize();
    cudaFree(d_result);
}

template <typename T>
typename std::enable_if<(sizeof(T) == 8)>::type get_class_centers(double *d_mat_seg, double *d_vals, 
                                                                  double *d_centers,
                                                                  int *d_scan_kcount, int *scan_kcount, 
                                                                  int *k_count, const size_t k, 
                                                                  const size_t MAT_COLS)
{
    double *d_result;
    CHECK(cudaMalloc((void**)&d_result, sizeof(double)*k*MAT_COLS*MAT_COLS)); 
    cudaMemset(d_result, 0, sizeof(double)*k*MAT_COLS*MAT_COLS);

    cudaStream_t *streams = (cudaStream_t*)malloc(k*sizeof(cudaStream_t));
    cublasHandle_t *handle = (cublasHandle_t*)malloc(k*sizeof(cublasHandle_t));
    cusolverDnHandle_t *cusolverH = (cusolverDnHandle_t*)malloc(k*sizeof(cusolverDnHandle_t));

    cublasStatus_t blasStat;
    cudaError_t streamStat;
    cusolverStatus_t solverStat;

    for(size_t i=0; i<k; i++)
    {
         blasStat = cublasCreate(&handle[i]);
         if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("shapeExtract CUBLAS initialization failed%d\n",blasStat);exit( -1 );}
         streamStat = cudaStreamCreate(&streams[i]);
         if (streamStat != cudaSuccess) {printf ("shapeExtract cudaStream initialization failed%d\n",solverStat);exit( -1 );}
         solverStat = cusolverDnCreate(&cusolverH[i]);
         if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("shapeExtract CUSOLVER initialization failed%d\n",solverStat);exit( -1 );}
    }
        
    double alpha = 1.0, beta = 0.0;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((MAT_COLS+threadsPerBlock.x-1)/threadsPerBlock.x, (MAT_COLS+threadsPerBlock.y-1)/threadsPerBlock.y);   

    for(size_t i=0; i<k; i++)
    {
        if(k_count[i] != 0)
        {
            blasStat = cublasSetStream(handle[i], streams[i]);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS stream set failed%d\n",blasStat);exit( -1 );}

            //yt*y
            blasStat = cublasDgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_T,   
                                   MAT_COLS, MAT_COLS, k_count[i],          
                                   &alpha, d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS,          
                                   d_mat_seg+MAT_COLS*scan_kcount[i], MAT_COLS, &beta,             
                                   d_result+i*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasDgemm failed%d\n",blasStat);exit( -1 );}

            T *d_matQ;
            CHECK(cudaMalloc((void**)&d_matQ, sizeof(T)*MAT_COLS*MAT_COLS));
            get_matQ<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(d_matQ, MAT_COLS, MAT_COLS);
            cudaDeviceSynchronize();

            //Q*S
            blasStat = cublasDgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N,   
                                   MAT_COLS, MAT_COLS, MAT_COLS,          
                                   &alpha, d_result+i*MAT_COLS*MAT_COLS, MAT_COLS,          
                                   d_matQ, MAT_COLS, &beta,             
                                   d_result+i*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasDgemm failed%d\n",blasStat);exit( -1 );}
       
            //(Q*S)*Q
            blasStat = cublasDgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N,   
                                   MAT_COLS, MAT_COLS, MAT_COLS,          
                                   &alpha, d_matQ, MAT_COLS,          
                                   d_result+i*MAT_COLS*MAT_COLS, MAT_COLS, &beta,             
                                   d_result+i*MAT_COLS*MAT_COLS, MAT_COLS);
            if (blasStat != CUBLAS_STATUS_SUCCESS) {printf ("CUBLAS cublasDgemm failed%d\n",blasStat);exit( -1 );}
       
            cudaFree(d_matQ);


            solverStat = cusolverDnSetStream(cusolverH[i], streams[i]);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER set stream failed%d\n",solverStat);exit( -1 );}
            double *d_EigVal; 
            double *d_work;
            int *devInfo; 
	        int lwork = 0;

            CHECK(cudaMalloc((void**)&d_EigVal, sizeof(double)*MAT_COLS));
            CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
	        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	        solverStat = cusolverDnDsyevd_bufferSize(cusolverH[i], jobz, uplo, MAT_COLS, 
                                                     d_result+i*MAT_COLS*MAT_COLS, MAT_COLS, 
                                                     d_EigVal, &lwork);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER set buffer failed%d\n",solverStat);exit( -1 );}
            
	        CHECK(cudaMalloc((void**)&d_work, sizeof(double)*lwork));

            solverStat = cusolverDnDsyevd(cusolverH[i], jobz, uplo, MAT_COLS, 
                                          d_result+i*MAT_COLS*MAT_COLS, MAT_COLS, 
                                          d_EigVal, d_work, lwork, devInfo);
            if (solverStat != CUSOLVER_STATUS_SUCCESS) {printf ("CUSOLVER cusolverDnDsyevd failed%d\n",solverStat);exit( -1 );}
            
            cudaFree(d_EigVal);
            cudaFree(d_work);
            cudaFree(devInfo);
        }
        
        cublasDestroy(handle[i]);
        cusolverDnDestroy(cusolverH[i]);
        cudaStreamDestroy(streams[i]);
       
    }
    
    const size_t blockSize = 256;
    const size_t gridSize  = (k+blockSize-1)/blockSize;
    get_center<<<gridSize, blockSize>>>(d_vals, d_result, d_centers, 
                                        d_scan_kcount, k, MAT_COLS);
    cudaDeviceSynchronize();
    cudaFree(d_result);
}

template<typename T>
void extract_shape(int *d_idx, T *d_mat, int *d_k_count, T *d_centers,
                   const size_t k, const size_t idx_len, const size_t mat_row,
                   const size_t mat_col)
{
    //统计每个类别数目
    if(idx_len < 2e6)
    {
        class_count<int><<<(idx_len + 256 -1)/256, 256>>>(d_idx, idx_len, d_k_count);
        cudaDeviceSynchronize();
    }
    else
    {
        int *d_count_tmp;
        const size_t step_size = 1000;
        const size_t stride = (idx_len%step_size == 0) ? (idx_len/step_size) : (idx_len/step_size+1);
        CHECK(cudaMalloc((void**)&d_count_tmp, sizeof(int)*stride*k));
        histo_split<<<idx_len/step_size/128+1, 128>>>(d_idx, d_count_tmp, step_size, k, idx_len);
        cudaDeviceSynchronize();
        histo_merge<<<idx_len/step_size/128+1, 128>>>(d_idx, d_count_tmp, d_k_count, step_size, k, idx_len); 
        cudaDeviceSynchronize();
        cudaFree(d_count_tmp);
    }

    int *d_loc2;
    int *count_k_cur;
    int *d_scan_kcount;
    CHECK(cudaMalloc((void**)&d_loc2, sizeof(int)*idx_len));
    CHECK(cudaMalloc((void**)&count_k_cur, sizeof(int)*k));
    CHECK(cudaMalloc((void**)&d_scan_kcount, sizeof(int)*k));
    cudaMemset(count_k_cur, 0, sizeof(int)*k);

    //求二级hash地址
    //小批量数据
    if(mat_row < 1e6)
    {
        get_loc2<<<1, 1>>>(d_idx, count_k_cur, d_loc2, idx_len);
        cudaDeviceSynchronize();
    }
    else
    {
        //大批量数据
        int *d_count_tmp;
        const size_t step_size = 1000;
        const size_t stride = (mat_row%step_size == 0) ? (mat_row/step_size) : (mat_row/step_size+1);
        CHECK(cudaMalloc((void**)&d_count_tmp, sizeof(int)*stride*k));
        loc2_split<<<mat_row/step_size/128+1, 128>>>(d_idx, d_count_tmp, d_loc2, step_size, k, mat_row);
        cudaDeviceSynchronize();
        per_scan<<<(k+32-1)/32, 32>>>(d_count_tmp, step_size, k, mat_row);
        cudaDeviceSynchronize();
        loc2_merge<<<mat_row/step_size/128+1, 128>>>(d_idx, d_count_tmp, d_loc2, step_size, k, mat_row);
    
        cudaDeviceSynchronize();
        cudaFree(d_count_tmp);
    }

    
    thrust::exclusive_scan(thrust::device, d_k_count, d_k_count + k, d_scan_kcount, 0);
    cudaDeviceSynchronize();
    

    T *d_mat_seg;
    CHECK(cudaMalloc((void**)&d_mat_seg, sizeof(T)*mat_row*mat_col));
   
    //地址散列
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((mat_row+threadsPerBlock.x-1)/threadsPerBlock.x, 
                   (mat_col+threadsPerBlock.y-1)/threadsPerBlock.y);
    block_alloc2D<<<numBlocks, threadsPerBlock>>>(d_mat_seg, d_mat, d_idx, d_scan_kcount,
                                                  d_loc2, mat_row, mat_col);
    cudaDeviceSynchronize();

    int *k_count = (int*)malloc(sizeof(int)*k); 
    int *scan_kcount = (int*)malloc(sizeof(int)*k);
    cudaMemcpy(k_count, d_k_count, sizeof(int)*k, cudaMemcpyDeviceToHost);
    cudaMemcpy(scan_kcount, d_scan_kcount, sizeof(int)*k, cudaMemcpyDeviceToHost);

    //相似度序列偏移
    for (size_t i = 0; i < k; i++)
    {
        T center_sum = thrust::reduce(thrust::device, d_centers+mat_col*i, d_centers+mat_col*(i+1), 0);
        cudaDeviceSynchronize();
        center_sum = center_sum>0 ? center_sum : (-1)*center_sum;
        size_t cur_count = k_count[i];
        if(center_sum > 1e-16 && cur_count != 0)
        {
            sbd3D(d_mat_seg+scan_kcount[i]*mat_col, d_centers+mat_col*i, d_mat_seg+scan_kcount[i]*mat_col, cur_count, mat_col);
        }
    }

    T *d_mat_val;
    CHECK(cudaMalloc((void**)&d_mat_val, sizeof(T)*k*mat_col));
    val_copy<<<(k+256-1)/256, 256>>>(d_mat_seg, d_mat_val, d_scan_kcount, k, mat_col);
    cudaDeviceSynchronize();

    z_norm_gpu_x(d_mat_seg, mat_row, mat_col, 1);

    //clacute maxium eigenvector for cluster center
    get_class_centers<T>(d_mat_seg, d_mat_val, d_centers, 
                         d_scan_kcount, scan_kcount, k_count, 
                         k, mat_col);
    
    
    free(k_count);
    free(scan_kcount);
    cudaFree(d_mat_val);
    cudaFree(d_mat_seg);
    cudaFree(d_scan_kcount);
    cudaFree(d_loc2);
    cudaFree(count_k_cur);
    
}


#endif