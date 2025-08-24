#include "infiniccl_test.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <pthread.h>
#include <vector>

#define TEST_INFINI(API__) CHECK_API_OR(API__, INFINI_STATUS_SUCCESS, return 1)
#define TEST_INFINI_THREAD(API__) CHECK_API_OR(API__, INFINI_STATUS_SUCCESS, return nullptr)

const size_t MAX_COUNT = 100ULL * 1024 * 1024;

const size_t TEST_COUNTS[] = {
    128,
    1024,
    4 * 1024,
    MAX_COUNT,
};

const infiniDtype_t TEST_DTYPES[] = {INFINI_DTYPE_F32, INFINI_DTYPE_F16};

const size_t WARM_UPS = 10;

const size_t ITERATIONS = 100;

struct ThreadArgs {
    int rank;
    int device_id;
    infinicclComm_t comm;
    infiniDevice_t device_type;
    infiniDtype_t dtype;
    size_t count;
    const void *data;
    const void *ans;
    int *result;
    double *time;
};

void setData(infiniDtype_t dtype, void *data, size_t count, float val) {
    switch (dtype) {
    case INFINI_DTYPE_F32:
        for (size_t i = 0; i < count; i++) {
            ((float *)data)[i] = val;
        }
        break;

    case INFINI_DTYPE_F16:
        for (size_t i = 0; i < count; i++) {
            ((fp16_t *)data)[i] = utils::cast<fp16_t>(val);
        }
        break;
    default:
        std::abort();
        break;
    }
}

template <typename T>
int checkData(const T *actual_, const T *expected_, size_t count) {
    int failed = 0;
    for (size_t i = 0; i < count; i++) {
        if constexpr (std::is_same<T, fp16_t>::value) {
            float actual = utils::cast<float>(actual_[i]);
            float expected = utils::cast<float>(expected_[i]);
            if (std::abs(actual - expected) > 1e-4) {
                failed += 1;
            }
        } else {
            if (std::abs(actual_[i] - expected_[i]) > 1e-4) {
                failed += 1;
            }
        }
    }
    return failed;
}

int checkData(const void *actual, const void *expected, infiniDtype_t dtype, size_t count) {
    switch (dtype) {
    case INFINI_DTYPE_F32:
        return checkData((const float *)actual, (const float *)expected, count);
    case INFINI_DTYPE_F16:
        return checkData((const fp16_t *)actual, (const fp16_t *)expected, count);
    default:
        std::abort();
        return 1;
    }
}

void *testAllReduceThread(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    *(args->result) = 1;
    TEST_INFINI_THREAD(infinirtSetDevice(args->device_type, args->device_id));
    infinirtStream_t stream;
    TEST_INFINI_THREAD(infinirtStreamCreate(&stream));
    void *output = std::malloc(args->count * infiniSizeOf(args->dtype));
    std::memset(output, 0, args->count * infiniSizeOf(args->dtype));
    void *buf;
    TEST_INFINI_THREAD(infinirtMalloc(&buf, args->count * infiniSizeOf(args->dtype)));
    TEST_INFINI_THREAD(infinirtMemcpy(buf, args->data, args->count * infiniSizeOf(args->dtype), INFINIRT_MEMCPY_H2D));
    TEST_INFINI_THREAD(infinicclAllReduce(buf, buf, args->count, args->dtype, INFINICCL_SUM, args->comm, stream));
    TEST_INFINI_THREAD(infinirtDeviceSynchronize());
    TEST_INFINI_THREAD(infinirtMemcpy(output, buf, args->count * infiniSizeOf(args->dtype), INFINIRT_MEMCPY_D2H));

    if (checkData(output, args->ans, args->dtype, args->count) != 0) {
        std::free(output);
        infinirtFree(buf);
        infinirtStreamDestroy(stream);
        return nullptr;
    }
    for (size_t i = 0; i < WARM_UPS; i++) {
        TEST_INFINI_THREAD(infinicclAllReduce(buf, buf, args->count, args->dtype, INFINICCL_SUM, args->comm, stream));
    }
    TEST_INFINI_THREAD(infinirtDeviceSynchronize());

    // measure time
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; i++) {
        TEST_INFINI_THREAD(infinicclAllReduce(buf, buf, args->count, args->dtype, INFINICCL_SUM, args->comm, stream));
    }
    TEST_INFINI_THREAD(infinirtDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    *args->time = elapsed_ms / ITERATIONS;

    *args->result = 0;

    std::free(output);
    infinirtFree(buf);
    infinirtStreamDestroy(stream);
    return nullptr;
}

int testAllReduce(infiniDevice_t device_type, int ndevice) {
    std::vector<ThreadArgs> thread_args(ndevice);
    std::vector<infinicclComm_t> comms(ndevice);
    std::vector<pthread_t> threads(ndevice);
    std::vector<int> device_ids(ndevice);
    std::vector<int> results(ndevice);
    std::vector<double> times(ndevice);
    void *data = std::malloc(MAX_COUNT * sizeof(float)); // Use float as max dtype size
    void *ans = std::malloc(MAX_COUNT * sizeof(float));

    for (int i = 0; i < ndevice; i++) {
        device_ids[i] = i;
    }
    TEST_INFINI(infinicclCommInitAll(device_type, comms.data(), ndevice, device_ids.data()));

    for (infiniDtype_t dtype : TEST_DTYPES) {
        setData(dtype, data, MAX_COUNT, 1.0f);
        setData(dtype, ans, MAX_COUNT, 1.0f * ndevice);
        for (size_t count : TEST_COUNTS) {
            std::cout << "Testing AllReduce with " << count << " elements of " << infiniDtypeToString(dtype) << std::endl;
            for (int rank = 0; rank < ndevice; rank++) {
                thread_args[rank] = {rank, device_ids[rank], comms[rank], device_type, dtype, count, data, ans, &results[rank], &times[rank]};
                pthread_create(&threads[rank], NULL, testAllReduceThread, &thread_args[rank]);
            }
            for (int rank = 0; rank < ndevice; rank++) {
                pthread_join(threads[rank], NULL);
            }
            int failed = std::accumulate(results.begin(), results.end(), 0);
            for (int rank = 0; rank < ndevice; rank++) {
                if (results[rank] != 0) {
                    std::cout << "Rank " << rank << ": incorrect results." << std::endl;
                } else {
                    std::cout << "Rank " << rank << ": " << times[rank] << " ms." << std::endl;
                }
            }

            if (failed > 0) {
                std::cout << "Failed with " << failed << " errors." << std::endl
                          << std::endl;
                std::free(data);
                std::free(ans);
                return 1;
            }
            std::cout << std::endl;
        }
    }

    std::free(data);
    std::free(ans);
    return 0;
}
