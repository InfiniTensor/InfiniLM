#include "utils_test.h"
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

void incrementOffset(ptrdiff_t &offset_1, const std::vector<ptrdiff_t> &strides_1, size_t data_size_1,
                     ptrdiff_t &offset_2, const std::vector<ptrdiff_t> &strides_2, size_t data_size_2,
                     std::vector<size_t> &counter, const std::vector<size_t> &shape) {
    for (ptrdiff_t d = shape.size() - 1; d >= 0; d--) {
        counter[d] += 1;
        offset_1 += strides_1[d] * data_size_1;
        offset_2 += strides_2[d] * data_size_2;
        if (counter[d] < shape[d]) {
            break;
        }
        counter[d] = 0;
        offset_1 -= shape[d] * strides_1[d] * data_size_1;
        offset_2 -= shape[d] * strides_2[d] * data_size_2;
    }
}

template <typename T>
size_t check_equal(
    const void *a,
    const void *b,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &strides_a,
    const std::vector<ptrdiff_t> &strides_b) {
    auto element_size = sizeof(T);
    std::vector<size_t> counter(shape.size(), 0);
    ptrdiff_t offset_a = 0;
    ptrdiff_t offset_b = 0;
    size_t numel = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    size_t fails = 0;
    for (size_t i = 0; i < numel; i++) {
        const T *ptr_a = reinterpret_cast<const T *>((const char *)a + offset_a);
        const T *ptr_b = reinterpret_cast<const T *>((const char *)b + offset_b);
        if (memcmp(ptr_a, ptr_b, element_size) != 0) {
            std::cerr << "Error at " << i << ": " << *ptr_a << " vs " << *ptr_b << std::endl;
            fails++;
        }
        incrementOffset(offset_a, strides_a, element_size, offset_b, strides_b, element_size, counter, shape);
    }
    return fails;
}

int test_transpose_any(size_t index, std::vector<size_t> shape, std::vector<ptrdiff_t> strides_a, std::vector<ptrdiff_t> strides_b) {
    auto numel = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    std::vector<float> a(numel);
    std::vector<float> b(numel);
    for (size_t i = 0; i < numel; i++) {
        a[i] = (float)i / numel;
    }

    utils::rearrange(b.data(), a.data(), shape.data(), strides_b.data(), strides_a.data(), shape.size(), sizeof(float));
    auto fails = check_equal<float>(a.data(), b.data(), shape, strides_a, strides_b);
    if (fails > 0) {
        std::cout << "test_transpose " << index << " failed" << std::endl;
        return 1;
    } else {
        std::cout << "test_transpose " << index << " passed" << std::endl;
        return 0;
    }
}

int test_rearrange() {
    return test_transpose_any(1, {3, 5}, {5, 1}, {1, 3})
         + test_transpose_any(2, {1, 2048}, {2048, 1}, {2048, 1})
         + test_transpose_any(3, {2, 2, 2, 4}, {16, 8, 1, 2}, {16, 8, 4, 1})
         + test_transpose_any(4, {2, 2, 2, 2, 4}, {32, 16, 8, 1, 2}, {32, 16, 8, 4, 1});
}
