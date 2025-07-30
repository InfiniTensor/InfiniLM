#include "test.h"
#include <cstring>
#include <infinirt.h>
#include <iostream>

bool testMemcpy(infiniDevice_t device, int deviceId, size_t dataSize) {

    std::cout << "==============================================\n"
              << "Testing memcpy on Device ID: " << deviceId << "\n"
              << "==============================================" << std::endl;

    // 分配主机内存
    std::cout << "[Device " << deviceId << "] Allocating host memory: " << dataSize * sizeof(float) << " bytes" << std::endl;
    std::vector<float> hostData(dataSize, 1.23f);
    std::vector<float> hostCopy(dataSize, 0.0f);

    // 分配设备内存
    void *deviceSrc = nullptr, *deviceDst = nullptr;
    size_t dataSizeInBytes = dataSize * sizeof(float);

    std::cout << "[Device " << deviceId << "] Allocating device memory: " << dataSizeInBytes << " bytes" << std::endl;
    if (infinirtMalloc(&deviceSrc, dataSizeInBytes) != INFINI_STATUS_SUCCESS) {
        std::cerr << "[Device " << deviceId << "] Failed to allocate device memory for deviceSrc." << std::endl;
        return false;
    }

    if (infinirtMalloc(&deviceDst, dataSizeInBytes) != INFINI_STATUS_SUCCESS) {
        std::cerr << "[Device " << deviceId << "] Failed to allocate device memory for deviceDst." << std::endl;
        infinirtFree(deviceSrc);
        return false;
    }

    // 复制数据到设备
    std::cout << "[Device " << deviceId << "] Copying data from host to device..." << std::endl;
    if (infinirtMemcpy(deviceSrc, hostData.data(), dataSizeInBytes, INFINIRT_MEMCPY_H2D) != INFINI_STATUS_SUCCESS) {
        std::cerr << "[Device " << deviceId << "] Failed to copy data from host to device." << std::endl;
        infinirtFree(deviceSrc);
        infinirtFree(deviceDst);
        return false;
    }

    // 设备内存间复制
    std::cout << "[Device " << deviceId << "] Copying data between device memory (D2D)..." << std::endl;
    if (infinirtMemcpy(deviceDst, deviceSrc, dataSizeInBytes, INFINIRT_MEMCPY_D2D) != INFINI_STATUS_SUCCESS) {
        std::cerr << "[Device " << deviceId << "] Failed to copy data from device to device." << std::endl;
        infinirtFree(deviceSrc);
        infinirtFree(deviceDst);
        return false;
    }

    // 设备数据复制回主机
    std::cout << "[Device " << deviceId << "] Copying data from device back to host..." << std::endl;
    if (infinirtMemcpy(hostCopy.data(), deviceDst, dataSizeInBytes, INFINIRT_MEMCPY_D2H) != INFINI_STATUS_SUCCESS) {
        std::cerr << "[Device " << deviceId << "] Failed to copy data from device to host." << std::endl;
        infinirtFree(deviceSrc);
        infinirtFree(deviceDst);
        return false;
    }

    // 数据验证
    std::cout << "[Device " << deviceId << "] Validating copied data..." << std::endl;
    if (std::memcmp(hostData.data(), hostCopy.data(), dataSizeInBytes) != 0) {
        std::cerr << "[Device " << deviceId << "] Data mismatch between hostData and hostCopy." << std::endl;
        infinirtFree(deviceSrc);
        infinirtFree(deviceDst);
        return false;
    }

    std::cout << "[Device " << deviceId << "] Data copied correctly!" << std::endl;

    // 释放设备内存
    std::cout << "[Device " << deviceId << "] Freeing device memory..." << std::endl;
    infinirtFree(deviceSrc);
    infinirtFree(deviceDst);

    std::cout << "[Device " << deviceId << "] Memory copy test PASSED!" << std::endl;

    return true;
}

bool testSetDevice(infiniDevice_t device, int deviceId) {

    std::cout << "Setting device " << device << " with ID: " << deviceId << std::endl;

    infiniStatus_t status = infinirtSetDevice(device, deviceId);

    if (status != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to set device " << device << " with ID " << deviceId << std::endl;
        return false;
    }

    return true;
}
