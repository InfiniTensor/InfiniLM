#include "tensor_aclnn.h"
#include "../../ops/utils.h"
#include <algorithm>

infiniopStatus_t aclnnTensorDescriptor::setDescriptor(aclDataType dtype, const std::vector<int64_t> &shape, const std::vector<int64_t> &strides) {
    if (shape.size() != strides.size()) {
        return STATUS_BAD_PARAM;
    }
    this->ndim = shape.size();
    this->shape = std::vector<int64_t>(shape);
    this->strides = std::vector<int64_t>(strides);
    this->dataType = dtype;

    // Set format
    // TODO: Support other format
    aclFormat format = aclFormat::ACL_FORMAT_ND;
    this->format = format;

    CHECK_STATUS(this->inferStorageShape(), STATUS_SUCCESS);

    return STATUS_SUCCESS;
}


/// @brief Infer storage shape. For now this ruturns a 1D shape of the total tensor storage size.
/// We don't see why higher dimensional storage shape is ever needed. To change if necesary.
infiniopStatus_t aclnnTensorDescriptor::inferStorageShape() {
    auto index = std::max_element(this->strides.begin(), this->strides.end());
    uint64_t max_stride_index = std::distance(this->strides.begin(), index);
    this->storageNdim = 1;
    this->storageShape = std::vector<int64_t>({this->shape[max_stride_index] * this->strides[max_stride_index]});

    return STATUS_SUCCESS;
}

/// @brief Set aclnnTensorDescriptor from infiniopTensorDescriptor
/// @param y infiniopTensorDescriptor
/// @return infiniopStatus_t
infiniopStatus_t aclnnTensorDescriptor::fromInfiniOpTensorDescriptor(infiniopTensorDescriptor_t y) {
    uint64_t ndim = y->ndim;
    // Cast shape type
    auto shape = std::vector<int64_t>(ndim);
    auto strides = std::vector<int64_t>(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        shape[i] = static_cast<int64_t>(y->shape[i]);
        strides[i] = y->strides[i];
    }
    return setDescriptor(toAclDataType(y->dt), shape, strides);
}

/// @brief Wrapper of aclCreateTensor. Create aclTensor.
/// See https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha001/apiref/appdevgapi/aclcppdevg_03_0168.html
/// @param desc Alias of aclnnTensorDescriptor*.
/// @param data Data ptr on device global mem.
/// @param tensor Pointer of pointer of aclTensor.
/// @return
infiniopStatus_t aclnnTensorDescriptor::createTensor(void *data) {
    if (this->t) {
        return STATUS_SUCCESS;
    }
    this->t = aclCreateTensor(this->shape.data(),
                              this->ndim,
                              this->dataType,
                              this->strides.data(),
                              this->offset,
                              this->format,
                              this->storageShape.data(),
                              this->storageNdim,
                              data);
    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnTensorDescriptor::destroyTensor() {
    auto ret = aclDestroyTensor(this->t);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclDesctroyTensor failed, ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);
    t = nullptr;

    return STATUS_SUCCESS;
}

aclnnTensorDescriptor::~aclnnTensorDescriptor() {
    if (this->t) {
        destroyTensor();
    }
}

/// @brief TensorDescriptor's string info
/// @param desc Alias of aclnnTensorDescriptor*.
/// @return String of aclnnTensorDescriptor.
char *aclnnTensorDescriptor::toString() {

    // Assume bufferSize
    size_t bufferSize = 1024 + this->ndim * 40 + this->storageNdim * 40;
    char *buffer = (char *) malloc(bufferSize);
    if (!buffer) return NULL;

    // Write info into buffer
    char *ptr = buffer;
    ptr += sprintf(ptr, "ndim: %" PRId64 "\n", this->ndim);

    ptr += sprintf(ptr, "shape: [");
    for (uint64_t i = 0; i < this->ndim; ++i) {
        ptr += sprintf(ptr, "%" PRId64, this->shape[i]);
        if (i < this->ndim - 1) {
            ptr += sprintf(ptr, ", ");
        }
    }
    ptr += sprintf(ptr, "]\n");

    ptr += sprintf(ptr, "stride: [");
    for (uint64_t i = 0; i < this->ndim; ++i) {
        ptr += sprintf(ptr, "%" PRId64, this->strides[i]);
        if (i < this->ndim - 1) {
            ptr += sprintf(ptr, ", ");
        }
    }
    ptr += sprintf(ptr, "]\n");

    ptr += sprintf(ptr, "offset: %" PRId64 "\n", this->offset);
    ptr += sprintf(ptr, "dataType: %s\n", dataTypeToString(this->dataType));
    ptr += sprintf(ptr, "format: %s\n", formatToString(this->format));

    ptr += sprintf(ptr, "storageShape: [");
    for (int64_t i = 0; i < this->storageNdim; ++i) {
        ptr += sprintf(ptr, "%" PRId64, this->storageShape[i]);
        if (i < this->storageNdim - 1) {
            ptr += sprintf(ptr, ", ");
        }
    }
    ptr += sprintf(ptr, "]\n");

    ptr += sprintf(ptr, "storageNdim: %" PRId64 "\n", this->storageNdim);

    return buffer;
}
