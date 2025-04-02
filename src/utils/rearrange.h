#ifndef __INFINIUTILS_REARRANGE_H__
#define __INFINIUTILS_REARRANGE_H__

#include "result.hpp"
#include <cstddef>
#include <vector>

namespace utils {

class RearrangeMeta {
    std::vector<ptrdiff_t> _meta;
    RearrangeMeta(std::vector<ptrdiff_t>);

public:
    static Result<RearrangeMeta> create(
        const size_t *shape,
        const ptrdiff_t *dst_strides,
        const ptrdiff_t *src_strides,
        size_t ndim,
        size_t element_size);

    size_t ndim() const;
    size_t unit() const;
    size_t count() const;

    const ptrdiff_t *idx_strides() const;
    const ptrdiff_t *dst_strides() const;
    const ptrdiff_t *src_strides() const;

    void launch(void *dst, const void *src) const;
};

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
