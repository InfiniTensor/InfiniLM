#include "common_cuda.cuh"

infiniStatus_t createCudaHandle(infiniopCudaHandle_t *handle_ptr, infiniDevice_t cuda_device_type) {
    // Create a new cublas handle pool
    int device_id = 0;
    CHECK_CUDA_OR_RETURN(cudaGetDevice(&device_id), INFINI_STATUS_DEVICE_NOT_INITIALIZED);
    auto pool = std::make_shared<Pool<cublasHandle_t>>();
    cublasHandle_t handle;
    cublasCreate(&handle);
    pool->push(std::move(handle));

    // create a cudnn handle pool
    auto cudnn_pool = std::make_shared<Pool<cudnnHandle_t>>();
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    cudnn_pool->push(std::move(cudnn_handle));

    // set CUDA device property
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // set device compute capability numbers
    int capability_major;
    int capability_minor;
    cudaDeviceGetAttribute(&capability_major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&capability_minor, cudaDevAttrComputeCapabilityMinor, device_id);

    *handle_ptr = new InfiniopCudaHandle{
        cuda_device_type,
        device_id,
        std::move(pool),
        std::move(cudnn_pool),
        std::move(prop),
        capability_major,
        capability_minor,
    };

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t destroyCudaHandle(infiniopCudaHandle_t handle_ptr) {
    handle_ptr->cublas_handle_pool = nullptr;
    handle_ptr->cudnn_handle_pool = nullptr;
    delete handle_ptr;

    return INFINI_STATUS_SUCCESS;
}
