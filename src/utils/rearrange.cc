#include "rearrange.h"
#include "check.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace utils {

void rearrange(
    void *dst_,
    const void *src_,
    const size_t *shape,
    const ptrdiff_t *dst_strides_,
    const ptrdiff_t *src_strides_,
    size_t ndim,
    size_t element_size) {

    struct Dim {
        size_t len;
        ptrdiff_t dst, src;
    };

    std::vector<Dim> dims;
    for (size_t i = 0; i < ndim; ++i) {
        // 剔除初始的 1 长维度
        if (shape[i] != 1) {
            auto sd = dst_strides_[i], ss = src_strides_[i];
            // assert (sd != 0)
            dims.push_back(Dim{shape[i], sd, ss});
        }
    }
    // 排序
    std::sort(dims.begin(), dims.end(), [](const Dim &a, const Dim &b) {
        if (std::abs(a.dst) == std::abs(b.dst)) {
            if (std::abs(a.src) == std::abs(b.src)) {
                return a.len < b.len;
            }
            return std::abs(a.src) > std::abs(b.src);
        }
        return std::abs(a.dst) > std::abs(b.dst);
    });
    // # 合并连续维度
    ptrdiff_t unit = element_size;
    // ## 合并末尾连续维度到 unit
    for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        if (it->dst == unit && it->src == unit) {
            unit *= it->len;
            ndim -= 1;
        } else {
            break;
        }
    }
    // ## 合并任意连续维度
    for (size_t i = ndim - 1; i > 0; --i) {
        auto &f = dims[i - 1];
        auto &b = dims[i];
        ptrdiff_t len = b.len;
        if (b.dst * len == f.dst && b.src * len == f.src) {
            f = Dim{b.len * f.len, b.dst, b.src};
            b = Dim{1, 0, 0};
            ndim -= 1;
        }
    }
    dims.resize(ndim);
    // 填写序号步长、输入步长和输出步长
    std::vector<ptrdiff_t>
        idx_strides(ndim + 1),
        dst_strides(ndim),
        src_strides(ndim);
    idx_strides[ndim] = 1;
    for (size_t i = 0; i < ndim; ++i) {
        idx_strides[i] = dims[i].len;
        dst_strides[i] = dims[i].dst;
        src_strides[i] = dims[i].src;
    }
    for (size_t i = ndim; i > 0; --i) {
        idx_strides[i - 1] *= idx_strides[i];
    }
    // 执行 rearrange
    if (idx_strides[0] == 1) {
        std::memcpy(dst_, src_, unit);
    } else {
        for (size_t i = 0; i < idx_strides[0]; ++i) {
            auto dst = reinterpret_cast<char *>(dst_);
            auto src = reinterpret_cast<const char *>(src_);
            for (size_t j = 0; j < ndim; ++j) {
                auto k = i / idx_strides[j + 1];
                dst += k * dst_strides[j];
                src += k * src_strides[j];
                i %= idx_strides[j + 1];
            }
            std::memcpy(dst, src, unit);
        }
    }
}

} // namespace utils
