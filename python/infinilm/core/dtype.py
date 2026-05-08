from infinilm.core.lib import _core


class dtype:
    def __init__(self, data_type):
        """An internal method. Please do not use this directly."""
        self._underlying = data_type

    def __repr__(self):
        repr_map = {
            _core.DataType.BYTE: "uint8",
            _core.DataType.BOOL: "bool",
            _core.DataType.I8: "int8",
            _core.DataType.I16: "int16",
            _core.DataType.I32: "int32",
            _core.DataType.I64: "int64",
            _core.DataType.U8: "uint8",
            _core.DataType.U16: "uint16",
            _core.DataType.U32: "uint32",
            _core.DataType.U64: "uint64",
            _core.DataType.F8: "float8",
            _core.DataType.F16: "float16",
            _core.DataType.F32: "float32",
            _core.DataType.F64: "float64",
            _core.DataType.C16: "complex16",
            _core.DataType.C32: "complex32",
            _core.DataType.C64: "complex64",
            _core.DataType.C128: "complex128",
            _core.DataType.BF16: "bfloat16",
        }
        return f"infinicore.{repr_map[self._underlying]}"

    def __eq__(self, other):
        """
        Compare two dtype objects for equality.

        Args:
            other: The object to compare with

        Returns:
            bool: True if both objects are dtype instances with the same underlying data type
        """
        if not isinstance(other, dtype):
            return False
        return self._underlying == other._underlying

    def __hash__(self):
        """
        Return a hash value for the dtype object.

        Returns:
            int: Hash value based on the underlying data type
        """
        return hash(self._underlying)


float32 = dtype(_core.DataType.F32)
float = float32
float64 = dtype(_core.DataType.F64)
double = float64
complex32 = dtype(_core.DataType.C32)
chalf = complex32
complex64 = dtype(_core.DataType.C64)
cfloat = complex64
complex128 = dtype(_core.DataType.C128)
cdouble = complex128
float16 = dtype(_core.DataType.F16)
half = float16
bfloat16 = dtype(_core.DataType.BF16)
uint8 = dtype(_core.DataType.U8)
int8 = dtype(_core.DataType.I8)
int16 = dtype(_core.DataType.I16)
short = int16
int32 = dtype(_core.DataType.I32)
int = int32
int64 = dtype(_core.DataType.I64)
long = int64
bool = dtype(_core.DataType.BOOL)
