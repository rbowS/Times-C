#ifndef GPU_PRINT__
#define GPU_PRINT__

//------------------print test--------------
template<typename T>
__global__ void print_test_f(T *d_mat2D, const size_t mat_row, const size_t mat_col)
{
    for (size_t i = 0; i < mat_row; i++)
    {
        for (size_t j = 0; j < mat_col; j++)
        {
            printf("%f ", d_mat2D[j + i*mat_col]);
        }
        printf("\n\n");
    }
    
}

template<typename T>
__device__ void d_print_test_f(T *d_mat2D, const size_t mat_row, const size_t mat_col)
{
    for (size_t i = 0; i < mat_row; i++)
    {
        for (size_t j = 0; j < mat_col; j++)
        {
            printf("%f ", d_mat2D[j + i*mat_col]);
        }
        printf("\n\n");
    }
    
}


template<typename T>
__global__ void print_test_d(T *d_mat2D, const size_t mat_row, const size_t mat_col)
{
    for (size_t i = 0; i < mat_row; i++)
    {
        for (size_t j = 0; j < mat_col; j++)
        {
            printf("%lf ", d_mat2D[j + i*mat_col]);
        }
        printf("\n\n");
    }
    
}

template<typename T>
__global__ void print_test_i(T *d_mat2D, const size_t mat_row, const size_t mat_col)
{
    for (size_t i = 0; i < mat_row; i++)
    {
        for (size_t j = 0; j < mat_col; j++)
        {
            printf("%d ", d_mat2D[j + i*mat_col]);
        }
        printf("\n\n");
    }
    
}

template<typename T>
__global__ void print_test_sumf(T *d_mat2D, const size_t mat_row, const size_t mat_col)
{
    float sum = 0;
    for (size_t i = 0; i < mat_row; i++)
    {
        sum = 0;
        for (size_t j = 0; j < mat_col; j++)
        {
            sum += d_mat2D[j + i*mat_col];
        
        }
        printf("%f ", sum);
        printf("\n\n");
    }
    
}


template<typename T>
__global__ void print_complx_f(T *d_mat2D, const size_t mat_row, const size_t mat_col)
{
    for (size_t i = 0; i < mat_row; i++)
    {
        for (size_t j = 0; j < mat_col; j++)
        {
            printf("%f+%fj ", d_mat2D[j + i*mat_col].x, d_mat2D[j + i*mat_col].y);
            //printf("%lf ", d_mat2D[j + i*mat_col].x);
        }
        printf("\n\n");
    }
    
}



template<typename T>
__global__ void print_complx_d(T *d_mat2D, const size_t mat_row, const size_t mat_col)
{
    for (size_t i = 0; i < mat_row; i++)
    {
        for (size_t j = 0; j < mat_col; j++)
        {
            printf("%lf+%lfj ", d_mat2D[j + i*mat_col].x, d_mat2D[j + i*mat_col].y);
            //printf("%lf ", d_mat2D[j + i*mat_col].x);
        }
        printf("\n\n");
    }
    
}
//------------------END---------------


#endif