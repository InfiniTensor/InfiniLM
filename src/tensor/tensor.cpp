#include "../tensor.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>

std::shared_ptr<TensorDesc>
TensorDesc::create(infiniDtype_t dtype, const std::vector<size_t> &shape,
                   const std::vector<ptrdiff_t> &strides) {
    return std::shared_ptr<TensorDesc>(new TensorDesc(dtype, shape, strides));
}

std::shared_ptr<TensorDesc>
TensorDesc::create(infiniDtype_t dtype, const std::vector<size_t> &shape) {
    auto ndim = shape.size();
    auto strides = std::vector<ptrdiff_t>(ndim);
    if (ndim > 0) {
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return create(dtype, shape, strides);
}

std::shared_ptr<TensorDesc>
TensorDesc::createWithOrder(infiniDtype_t dtype, const std::vector<size_t> &shape,
                            const std::vector<size_t> &order) {
    ASSERT_EQ(shape.size(), order.size());
    auto ndim = shape.size();
    if (ndim == 0) {
        return create(dtype, shape);
    }
    auto strides = std::vector<ptrdiff_t>(order.size());
    auto idx = std::find(order.begin(), order.end(), size_t(ndim - 1));
    strides[std::distance(order.begin(), idx)] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        auto prev_dim = shape[std::distance(order.begin(), idx)];
        auto prev_stride = strides[std::distance(order.begin(), idx)];
        idx = std::find(order.begin(), order.end(), size_t(i));
        strides[std::distance(order.begin(), idx)] = prev_stride * prev_dim;
    }
    return create(dtype, shape, strides);
}

infiniopTensorDescriptor_t TensorDesc::desc() const {
    if (_desc == nullptr) {
        RUN_INFINI(infiniopCreateTensorDescriptor(
            (infiniopTensorDescriptor_t *)(&_desc), _shape.size(), _shape.data(),
            _strides.data(), _dtype));
    }
    return _desc;
};

void TensorDesc::resetDesc() {
    if (this->_desc != nullptr) {
        infiniopDestroyTensorDescriptor(this->_desc);
        this->_desc = nullptr;
    }
}

void TensorDesc::computeTensorDesHash() {
    _seed = 0;
    for (auto dim : this->shape()) {
        hash_combine(_seed, dim);
    }
    for (auto stride : this->strides()) {
        hash_combine(_seed, static_cast<size_t>(stride));
    }
}

bool TensorDesc::isContigous() const {
    auto ndim = this->ndim();
    auto shape = this->shape();
    auto strides = std::vector<ptrdiff_t>(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    ASSERT_EQ(strides.size(), this->_strides.size());
    return std::equal(strides.begin(), strides.end(), this->_strides.begin());
}

std::string TensorDesc::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

TensorDesc::~TensorDesc() {
    this->resetDesc();
}

const std::vector<size_t> &Tensor::shape() const { return this->_desc->shape(); }
const std::vector<ptrdiff_t> &Tensor::strides() const { return this->_desc->strides(); }
size_t Tensor::ndim() const { return this->_desc->ndim(); }
infiniDtype_t Tensor::dtype() const { return this->_desc->dtype(); }
infiniDevice_t Tensor::deviceType() const { return this->_storage->deviceType(); }
int Tensor::deviceId() const { return this->_storage->deviceId(); }
Tensor::~Tensor() {}

ptrdiff_t Tensor::dataOffset() const {
    return _offset;
}

infiniopTensorDescriptor_t Tensor::desc() const { return _desc->desc(); }

std::shared_ptr<Tensor> Tensor::buffer(infiniDtype_t dtype,
                                       const std::vector<size_t> &shape,
                                       std::shared_ptr<MemoryPool> pool) {
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
    auto ndim = shape.size();

    size_t size = std::accumulate(shape.begin(), shape.end(), dsize(dtype), std::multiplies<size_t>());
    auto strides = std::vector<ptrdiff_t>(ndim);
    if (ndim > 0) {
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    tensor->_storage = Storage::createFromPool(size, pool);
    tensor->_desc = TensorDesc::create(dtype, shape, strides);
    tensor->_offset = 0;
    return tensor;
}

std::shared_ptr<Tensor> Tensor::weight(void *data, infiniDtype_t dtype,
                                       const std::vector<size_t> &shape) {
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
    auto ndim = shape.size();
    size_t size = std::accumulate(shape.begin(), shape.end(), dsize(dtype), std::multiplies<size_t>());
    auto strides = std::vector<ptrdiff_t>(ndim);
    if (ndim > 0) {
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    tensor->_storage = Storage::create(size);
    tensor->_desc = TensorDesc::create(dtype, shape, strides);
    // NOTE: 为兼容部分平台（沐曦）多线程并发对同一host数据执行memcpy卡死问题
    static std::mutex mutex;
    {
        std::lock_guard<std::mutex> lock(mutex);
        RUN_INFINI(infinirtMemcpy(tensor->_storage->memory(),
                                  data, size, INFINIRT_MEMCPY_H2D));
    }

    tensor->_offset = 0;
    return tensor;
}

std::shared_ptr<Tensor> Tensor::memShare(const std::vector<size_t> &shape, infiniDtype_t dtype_) const {
    auto dtype = dtype_ == INFINI_DTYPE_INVALID ? this->dtype() : dtype_;
    size_t size = std::accumulate(shape.begin(), shape.end(), dsize(dtype), std::multiplies<size_t>());
    ASSERT(size <= this->_storage->size());

    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
    auto ndim = shape.size();
    auto strides = std::vector<ptrdiff_t>(ndim);
    if (ndim > 0) {
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    tensor->_storage = this->_storage;
    tensor->_offset = 0;
    tensor->_desc = TensorDesc::create(dtype, shape, strides);
    return tensor;
}

void *Tensor::dataImpl(ptrdiff_t offset) const {
    return (char *)(this->_storage->memory()) + this->_offset + offset * dsize(this->dtype());
}

void *Tensor::data(ptrdiff_t offset) {
    return this->dataImpl(offset);
}

const void *Tensor::data(ptrdiff_t offset) const {
    return this->dataImpl(offset);
}

void Tensor::copyFrom(std::shared_ptr<Tensor const> src,
                      infiniopHandle_t handle, infinirtStream_t stream) {
    ASSERT_EQ(this->shape(), src->shape());
    ASSERT_EQ(this->dtype(), src->dtype());
    infiniopRearrangeDescriptor_t desc;
    RUN_INFINI(infiniopCreateRearrangeDescriptor(
        handle, &desc, this->desc(), src->desc()));
    RUN_INFINI(infiniopRearrange(desc, this->data(), src->data(),
                                 stream));
    RUN_INFINI(infiniopDestroyRearrangeDescriptor(desc));
}

bool Tensor::isContigous() const {
    return this->_desc->isContigous();
}

template <typename T>
void print_data(T *data, const std::vector<size_t> &shape,
                const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

template <>
void print_data(uint16_t const *data, const std::vector<size_t> &shape,
                const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << f16_to_f32(data[i * strides[dim]]) << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void print_data_bf16(uint16_t const *data, const std::vector<size_t> &shape,
                     const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << bf16_to_f32(data[i * strides[dim]]) << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << this->_desc->info()
       << " device=" << this->deviceType()
       << " device_id=" << this->deviceId();
    return this->_desc->info();
}

size_t Tensor::seed() const {
    return this->_desc->seed();
}

std::shared_ptr<Tensor> Tensor::view(const std::vector<size_t> &new_shape) const {
    // Calculate total number of elements
    size_t numel = 1;
    for (auto s : shape()) {
        numel *= s;
    }

    size_t new_numel = 1;
    for (auto s : new_shape) {
        new_numel *= s;
    }

    ASSERT(numel == new_numel);

    // Handle empty tensors
    if (numel == 0) {
        return this->view_as(new_shape, {});
    }

    // Special case: view(-1) flattens the tensor
    if (new_shape.size() == 1 && new_shape[0] == static_cast<size_t>(-1)) {
        std::vector<size_t> flat_shape = {numel};
        return this->view_as(flat_shape, {});
    }

    // Check for -1 in new_shape (infer dimension)
    std::vector<size_t> inferred_shape = new_shape;
    size_t infer_index = static_cast<size_t>(-1);
    size_t known_elements = 1;

    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == static_cast<size_t>(-1)) {
            ASSERT(infer_index == static_cast<size_t>(-1)); // Only one -1 allowed
            infer_index = i;
        } else {
            known_elements *= new_shape[i];
        }
    }

    if (infer_index != static_cast<size_t>(-1)) {
        ASSERT(numel % known_elements == 0);
        inferred_shape[infer_index] = numel / known_elements;
    }

    // For contiguous tensors, compute standard row-major strides
    if (this->isContigous()) {
        std::vector<ptrdiff_t> new_strides(inferred_shape.size());
        if (!inferred_shape.empty()) {
            new_strides.back() = 1;
            for (int i = static_cast<int>(inferred_shape.size()) - 2; i >= 0; --i) {
                new_strides[i] = new_strides[i + 1] * static_cast<ptrdiff_t>(inferred_shape[i + 1]);
            }
        }
        return this->view_as(inferred_shape, new_strides);
    }

    // For non-contiguous tensors
    std::vector<size_t> old_shape = shape();
    std::vector<ptrdiff_t> old_strides = strides();
    std::vector<ptrdiff_t> new_strides(inferred_shape.size(), 0);

    size_t old_idx = old_shape.size() - 1;
    size_t new_idx = inferred_shape.size() - 1;

    if (new_idx != static_cast<size_t>(-1)) {
        new_strides[new_idx] = 1;
    }

    while (old_idx != static_cast<size_t>(-1) && new_idx != static_cast<size_t>(-1)) {
        size_t old_size = old_shape[old_idx];
        size_t new_size = inferred_shape[new_idx];

        if (old_size == 1) {
            old_idx--;
        } else if (new_size == 1) {
            new_strides[new_idx] = (new_idx == inferred_shape.size() - 1) ? 1 : new_strides[new_idx + 1];
            new_idx--;
        } else if (old_size == new_size) {
            new_strides[new_idx] = old_strides[old_idx];
            old_idx--;
            new_idx--;
        } else if (old_size < new_size) {
            size_t combined_size = old_size;
            ptrdiff_t combined_stride = old_strides[old_idx];
            old_idx--;

            while (old_idx != static_cast<size_t>(-1) && combined_size < new_size) {
                ASSERT(static_cast<size_t>(old_strides[old_idx]) == old_shape[old_idx + 1] * static_cast<size_t>(old_strides[old_idx + 1]));
                combined_size *= old_shape[old_idx];
                combined_stride = old_strides[old_idx];
                old_idx--;
            }

            ASSERT(combined_size == new_size);
            new_strides[new_idx] = combined_stride;
            new_idx--;
        } else {
            size_t remaining_size = old_size / new_size;
            ASSERT(old_size % new_size == 0);
            new_strides[new_idx] = old_strides[old_idx] * static_cast<ptrdiff_t>(remaining_size);
            new_idx--;

            if (remaining_size != 1) {
                if (new_idx != static_cast<size_t>(-1)) {
                    inferred_shape[new_idx] = remaining_size;
                    new_strides[new_idx] = old_strides[old_idx];
                    new_idx--;
                } else {
                    ASSERT(false);
                }
            }
            old_idx--;
        }
    }

    // Fill remaining dimensions (must be size 1)
    while (new_idx != static_cast<size_t>(-1)) {
        ASSERT(inferred_shape[new_idx] == 1);
        new_strides[new_idx] = new_strides[new_idx + 1];
        new_idx--;
    }

    return this->view_as(inferred_shape, new_strides);
}

std::shared_ptr<Tensor> Tensor::view_as(const std::vector<size_t> &new_shape) const {
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
    tensor->_storage = this->_storage;
    tensor->_desc = TensorDesc::create(this->dtype(), new_shape);
    tensor->_offset = this->_offset;
    return tensor;
}

std::shared_ptr<Tensor> Tensor::view_as(const std::vector<size_t> &new_shape, const std::vector<ptrdiff_t> &new_strides) const {
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
    tensor->_storage = this->_storage;
    tensor->_desc = TensorDesc::create(this->dtype(), new_shape, new_strides);
    tensor->_offset = this->_offset;
    return tensor;
}

void Tensor::debug(const std::string &filename) const {
    RUN_INFINI(infinirtDeviceSynchronize());

    std::cout << info() << std::endl;

    void const *cpu_data;
    if (this->deviceType() != INFINI_DEVICE_CPU) {
        void *cpu_memory = std::malloc(this->_storage->size());
        RUN_INFINI(infinirtMemcpy(cpu_memory, this->_storage->memory(),
                                  this->_storage->size(), INFINIRT_MEMCPY_D2H));
        cpu_data = cpu_memory;
    } else {
        cpu_data = this->_storage->memory();
    }

    if (!filename.empty()) {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error opening file for writing: " << filename << "\n";
            return;
        }
        outFile.write(reinterpret_cast<const char *>(cpu_data), this->_storage->size());
        outFile.close();
        std::cout << "Data written to file: " << filename << "\n";
        return;
    }

    switch (this->dtype()) {
    case INFINI_DTYPE_F16:
        print_data((uint16_t const *)((char const *)cpu_data + dataOffset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_F32:
        print_data((float const *)((char const *)cpu_data + dataOffset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_U64:
        print_data((uint64_t const *)((char const *)cpu_data + dataOffset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_I64:
        print_data((int64_t const *)((char const *)cpu_data + dataOffset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_U32:
        print_data((uint32_t const *)((char const *)cpu_data + dataOffset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_I32:
        print_data((int32_t const *)((char const *)cpu_data + dataOffset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_BF16:
        print_data_bf16((uint16_t const *)((char const *)cpu_data + dataOffset()),
                        this->shape(), this->strides(), 0);
        break;
    default:
        PANIC("Unsupported data type");
    }
}

void Tensor::debug() const { this->debug(""); }
