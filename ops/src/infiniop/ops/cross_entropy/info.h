#ifndef CROSS_ENTROPY_INFO_H
#define CROSS_ENTROPY_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

#include <cstddef>

struct CrossEntropyInfo {
    int dtype;
    int target_dtype;
    size_t outer_size;
    size_t vocab_size;
    ptrdiff_t x_stride;
};

#endif
