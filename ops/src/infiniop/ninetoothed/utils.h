#ifndef __NINETOOTHED_UTILS__
#define __NINETOOTHED_UTILS__

#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

namespace ninetoothed {

template <typename T = float>
class Tensor {
public:
    using Data = decltype(NineToothedTensor::data);

    using Size = std::remove_pointer_t<decltype(NineToothedTensor::shape)>;

    using Stride = std::remove_pointer_t<decltype(NineToothedTensor::strides)>;

    template <typename Shape, typename Strides>
    Tensor(const void *data, Shape shape, Strides strides) : data_{data}, shape_{shape}, strides_{strides}, ndim_{shape_.size()} {}

    Tensor(const void *data, std::initializer_list<Size> shape, std::initializer_list<Stride> strides) : Tensor{data, decltype(shape_){shape}, decltype(strides_){strides}} {}

    Tensor(const void *data, const Size *shape, const Stride *strides, Size ndim) : data_{data}, shape_{shape, shape + ndim}, strides_{strides, strides + ndim}, ndim_{shape_.size()} {}

    Tensor(const T value) : value_{value}, data_{&value_}, ndim_{0} {}

    operator NineToothedTensor() { return {const_cast<Data>(data_), shape_.data(), strides_.data()}; }

    template <typename Shape>
    Tensor expand(const Shape &sizes) const {
        auto new_ndim{sizes.size()};

        decltype(shape_) shape(new_ndim, 1);
        decltype(strides_) strides(new_ndim, 0);

        auto num_new_dims{new_ndim - ndim_};

        for (auto dim{decltype(ndim_){0}}; dim < ndim_; ++dim) {
            shape[dim + num_new_dims] = shape_[dim];
            strides[dim + num_new_dims] = strides_[dim];
        }

        for (auto dim{decltype(new_ndim){0}}; dim < new_ndim; ++dim) {
            if (sizes[dim] == std::numeric_limits<std::remove_reference_t<decltype(sizes[dim])>>::max() || shape[dim] != 1) {
                continue;
            }

            shape[dim] = sizes[dim];
            strides[dim] = 0;
        }

        return {data_, shape, strides};
    }

    Tensor expand_as(const Tensor &other) const {
        return expand(other.shape_);
    }

private:
    const void *data_{nullptr};

    std::vector<Size> shape_;

    std::vector<Stride> strides_;

    Size ndim_{0};

    T value_{0};
};

} // namespace ninetoothed

#endif
