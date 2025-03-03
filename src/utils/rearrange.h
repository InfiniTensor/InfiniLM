#ifndef INFINIUTILS_REARRANGE_H
#define INFINIUTILS_REARRANGE_H
#include <stddef.h>

void rearrange(void *dst,
               const void *src,
               const size_t *shape,
               const ptrdiff_t *dst_strides,
               const ptrdiff_t *src_strides,
               const size_t ndim,
               size_t element_size);

#endif // INFINIUTILS_REARRANGE_H
