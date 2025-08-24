from ctypes import c_int, Structure, POINTER


class TensorDescriptor(Structure):
    _fields_ = []


infiniopTensorDescriptor_t = POINTER(TensorDescriptor)


class Handle(Structure):
    _fields_ = [("device", c_int), ("device_id", c_int)]


infiniopHandle_t = POINTER(Handle)


class OpDescriptor(Structure):
    _fields_ = [("device", c_int), ("device_id", c_int)]


infiniopOperatorDescriptor_t = POINTER(OpDescriptor)
