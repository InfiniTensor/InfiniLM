#include "./common_cuda.cuh"

infiniopStatus_t createCudaHandle(infiniopCudaHandle_t *handle_ptr, int device_id, infiniDevice_t cuda_device_type) {
    // Check if device_id is valid
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_id >= device_count) {
        return INFINIOP_STATUS_BAD_DEVICE;
    }

    // Create a new cublas handle pool
    auto pool = std::make_shared<Pool<cublasHandle_t>>();
    if (cudaSetDevice(device_id) != cudaSuccess) {
        return INFINIOP_STATUS_BAD_DEVICE;
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    pool->push(std::move(handle));

    // create a cudnn handle pool
    auto cudnn_pool = std::make_shared<Pool<cudnnHandle_t>>();
    cudnnHandle_t cudnn_handle;
    checkCudnnError(cudnnCreate(&cudnn_handle));
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

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t destroyCudaHandle(infiniopCudaHandle_t handle_ptr) {
    handle_ptr->cublas_handles_t = nullptr;
    handle_ptr->cudnn_handles_t = nullptr;
    delete handle_ptr;

    return INFINIOP_STATUS_SUCCESS;
}
