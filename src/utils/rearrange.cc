#include "rearrange.h"
#include "check.h"
#include <algorithm>
#include <cstring>
#include <vector>

#ifdef ENABLE_OMP
#include <omp.h>
#endif

namespace utils {

RearrangeMeta::RearrangeMeta(std::vector<ptrdiff_t> meta)
    : _meta(std::move(meta)) {}

Result<RearrangeMeta> RearrangeMeta::create(
    const size_t *shape,
    const ptrdiff_t *dst_strides_,
    const ptrdiff_t *src_strides_,
    size_t ndim,
    size_t element_size) {

    ptrdiff_t unit = element_size;

    struct Dim {
        size_t len;
        ptrdiff_t dst, src;
    };

    std::vector<Dim> dims;
    for (size_t i = 0; i < ndim; ++i) {
        // 剔除初始的 1 长维度
        if (shape[i] != 1) {
            auto sd = dst_strides_[i] * unit, ss = src_strides_[i] * unit;
            if (sd == 0) {
                return INFINI_STATUS_BAD_TENSOR_STRIDES;
            }
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
    ndim = dims.size();
    // # 合并连续维度
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
    for (ptrdiff_t i = ndim - 1; i > 0; --i) {
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
    std::vector<ptrdiff_t> meta(2 + ndim * 3);
    meta[0] = unit;
    meta[1 + ndim] = 1;
    for (size_t i = 0; i < ndim; ++i) {
        meta[1 + i] = dims[i].len;
        meta[1 + 1 + ndim + i] = dims[i].dst;
        meta[1 + 1 + ndim * 2 + i] = dims[i].src;
    }
    for (ptrdiff_t i = ndim; i > 0; --i) {
        meta[1 + i - 1] *= meta[1 + i];
    }
    return Result<RearrangeMeta>(meta);
}

size_t RearrangeMeta::ndim() const { return (_meta.size() - 2) / 3; }
size_t RearrangeMeta::unit() const { return _meta[0]; }
size_t RearrangeMeta::count() const { return _meta[1]; }

const ptrdiff_t *RearrangeMeta::idx_strides() const { return _meta.data() + 2; }
const ptrdiff_t *RearrangeMeta::dst_strides() const { return idx_strides() + ndim(); }
const ptrdiff_t *RearrangeMeta::src_strides() const { return dst_strides() + ndim(); }

void RearrangeMeta::launch(void *dst_, const void *src_) const {
    auto const ndim_ = ndim();
    auto const count_ = count();
    auto const unit_ = unit();
    auto const idx_strides_ = idx_strides();
    auto const dst_strides_ = dst_strides();
    auto const src_strides_ = src_strides();
    // 执行 rearrange
    if (count_ == 1) {
        std::memcpy(dst_, src_, unit_);
    } else {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < (ptrdiff_t)count_; ++i) {
            auto dst = reinterpret_cast<char *>(dst_);
            auto src = reinterpret_cast<const char *>(src_);
            auto rem = i;
            for (size_t j = 0; j < ndim_; ++j) {
                auto k = rem / idx_strides_[j];
                dst += k * dst_strides_[j];
                src += k * src_strides_[j];
                rem %= idx_strides_[j];
            }
            std::memcpy(dst, src, unit_);
        }
    }
}

void rearrange(
    void *dst,
    const void *src,
    const size_t *shape,
    const ptrdiff_t *dst_strides,
    const ptrdiff_t *src_strides,
    size_t ndim,
    size_t element_size) {

    auto scheme = RearrangeMeta::create(shape, dst_strides, src_strides, ndim, element_size);
    if (scheme) {
        scheme->launch(dst, src);
    } else {
        std::abort();
    }
}

utils::Result<RearrangeMeta> RearrangeMeta::distributeUnit(const std::vector<size_t> &candidates) const {
    // 获取当前的unit大小
    size_t current_unit = _meta[0];

    // 寻找满足条件的unit值：当前unit能被其整除
    size_t new_unit = 0;
    for (size_t candidate : candidates) {
        if (current_unit % candidate == 0) {
            new_unit = candidate;
            break;
        }
    }

    // 如果没找到合适的值，返回错误
    if (new_unit == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // 如果找到的值就是当前unit，返回自身的副本
    if (new_unit == current_unit) {
        return Result<RearrangeMeta>(_meta);
    }

    // 获取当前维度
    size_t ndim_value = this->ndim();

    // 创建新的布局数组
    std::vector<ptrdiff_t> layout(2 + (ndim_value + 1) * 3, 0);

    // 设置新的unit值
    layout[0] = new_unit;

    // 计算扩展因子
    ptrdiff_t extra = current_unit / new_unit;

    // 计算步长指针的偏移量
    ptrdiff_t idx_offset = 1;

    // 在新布局中设置相应的指针
    ptrdiff_t *new_idx = layout.data() + 1;
    ptrdiff_t *new_dst = layout.data() + 2 + (ndim_value + 1);
    ptrdiff_t *new_src = layout.data() + 2 + (ndim_value + 1) * 2;

    // 复制并调整索引步长

    // 索引步长需要重新计算
    // 首先复制原来的索引步长
    for (size_t i = 0; i < ndim_value + 1; ++i) {
        new_idx[i] = _meta[idx_offset + i] * extra;
    }

    // 设置最后一个维度的步长为1
    new_idx[ndim_value + 1] = 1;

    // 复制目标步长数据，并添加新单元大小
    for (size_t i = 0; i < ndim_value; ++i) {
        new_dst[i] = dst_strides()[i];
    }
    new_dst[ndim_value] = new_unit;

    // 复制源步长数据，并添加新单元大小
    for (size_t i = 0; i < ndim_value; ++i) {
        new_src[i] = src_strides()[i];
    }
    new_src[ndim_value] = new_unit;

    return Result<RearrangeMeta>(layout);
}

} // namespace utils
