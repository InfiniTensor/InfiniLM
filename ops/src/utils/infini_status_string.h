#ifndef INFINI_STATUS_STRING_H
#define INFINI_STATUS_STRING_H
#include <infinicore.h>

inline const char *infini_status_string(infiniStatus_t status) {
    switch (status) {
    case INFINI_STATUS_SUCCESS:
        return "Success";
    case INFINI_STATUS_INTERNAL_ERROR:
        return "Internal Error";
    case INFINI_STATUS_NOT_IMPLEMENTED:
        return "Not Implemented";
    case INFINI_STATUS_BAD_PARAM:
        return "Bad Parameter";
    case INFINI_STATUS_NULL_POINTER:
        return "Null Pointer";
    case INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED:
        return "Device Type Not Supported";
    case INFINI_STATUS_DEVICE_NOT_FOUND:
        return "Device Not Found";
    case INFINI_STATUS_DEVICE_NOT_INITIALIZED:
        return "Device Not Initialized";
    case INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED:
        return "Device Architecture Not Supported";
    case INFINI_STATUS_BAD_TENSOR_DTYPE:
        return "Bad Tensor Data Type";
    case INFINI_STATUS_BAD_TENSOR_SHAPE:
        return "Bad Tensor Shape";
    case INFINI_STATUS_BAD_TENSOR_STRIDES:
        return "Bad Tensor Strides";
    case INFINI_STATUS_INSUFFICIENT_WORKSPACE:
        return "Insufficient Workspace";
    default:
        return "Unknown Error";
    }
}

#endif /* INFINI_STATUS_STRING_H */
