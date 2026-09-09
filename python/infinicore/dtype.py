from infinicore.lib import _infinicore


class dtype:
    def __init__(self, underlying):
        self._underlying = underlying

    def __repr__(self):
        return _DTYPE_NAMES[self._underlying]

    def __eq__(self, other):
        return isinstance(other, dtype) and self._underlying == other._underlying

    def __hash__(self):
        return hash(self._underlying)


_DTYPE_NAMES = {
    _infinicore.DataType.INT8: "int8",
    _infinicore.DataType.INT16: "int16",
    _infinicore.DataType.INT32: "int32",
    _infinicore.DataType.INT64: "int64",
    _infinicore.DataType.UINT8: "uint8",
    _infinicore.DataType.UINT16: "uint16",
    _infinicore.DataType.UINT32: "uint32",
    _infinicore.DataType.UINT64: "uint64",
    _infinicore.DataType.FLOAT16: "float16",
    _infinicore.DataType.BFLOAT16: "bfloat16",
    _infinicore.DataType.FLOAT32: "float32",
    _infinicore.DataType.FLOAT64: "float64",
}

int8 = dtype(_infinicore.DataType.INT8)
int16 = dtype(_infinicore.DataType.INT16)
int32 = dtype(_infinicore.DataType.INT32)
int64 = dtype(_infinicore.DataType.INT64)
uint8 = dtype(_infinicore.DataType.UINT8)
uint16 = dtype(_infinicore.DataType.UINT16)
uint32 = dtype(_infinicore.DataType.UINT32)
uint64 = dtype(_infinicore.DataType.UINT64)
float16 = dtype(_infinicore.DataType.FLOAT16)
bfloat16 = dtype(_infinicore.DataType.BFLOAT16)
float32 = dtype(_infinicore.DataType.FLOAT32)
float64 = dtype(_infinicore.DataType.FLOAT64)

half = float16
float = float32
double = float64
short = int16
int = int32
long = int64
