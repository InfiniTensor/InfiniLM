#ifndef __INFINIUTILS_REARRANGE_H__
#define __INFINIUTILS_REARRANGE_H__

#include <stddef.h>

namespace utils {

void rearrange(
    void *dst,
    const void *src,
    const size_t *shape,
    const ptrdiff_t *dst_strides,
    const ptrdiff_t *src_strides,
    size_t ndim,
    size_t element_size);

} // namespace utils

#endif // __INFINIUTILS_REARRANGE_H__
