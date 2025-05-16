#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

template <typename T, typename U>
class RoPEKernel {
public:
    __aicore__ inline RoPEKernel() {}
    // Init op
    // pos position vector
    // x input tensor
    // y output tensor
    // tensor shape [nt, nh, dh]
    // make block_num = nh, tile_len = dh
    __aicore__ inline void init(GM_ADDR y,
                                GM_ADDR x,
                                GM_ADDR pos,
                                GM_ADDR sin,
                                GM_ADDR cos,
                                size_t dh,
                                ptrdiff_t st_ynt,
                                ptrdiff_t st_ynh,
                                ptrdiff_t st_xnt,
                                ptrdiff_t st_xnh);
    __aicore__ inline void process(size_t seq_len);

private:
    // Copy a tile into UB
    __aicore__ inline void copyIn(size_t i);
    __aicore__ inline void compute(size_t i);
    __aicore__ inline void copyOut(size_t i);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> sinQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> cosQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<TPosition::VECCALC> tmpOddBuf;
    TBuf<TPosition::VECCALC> tmpEvenBuf;
    TBuf<TPosition::VECCALC> tmpOddBuf1;
    TBuf<TPosition::VECCALC> tmpOddBuf2;
    TBuf<TPosition::VECCALC> tmpEvenBuf1;
    TBuf<TPosition::VECCALC> tmpEvenBuf2;

    GlobalTensor<T> xGm, yGm;
    GlobalTensor<U> pGm;
    GlobalTensor<T> sinGm;
    GlobalTensor<T> cosGm;

    size_t _block_idx;
    size_t _tile_len;
    size_t _copy_len;
    size_t _half_copy_len;

    // stridey[_st_ynt, _st_ynh, 1]
    ptrdiff_t _st_ynt;
    ptrdiff_t _st_ynh;
    // stridex[_st_xnt, _st_xnh, 1]
    ptrdiff_t _st_xnt;
    ptrdiff_t _st_xnh;
};

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::init(GM_ADDR y,
                                              GM_ADDR x,
                                              GM_ADDR pos,
                                              GM_ADDR sin,
                                              GM_ADDR cos,
                                              size_t dh,
                                              ptrdiff_t st_ynt,
                                              ptrdiff_t st_ynh,
                                              ptrdiff_t st_xnt,
                                              ptrdiff_t st_xnh) {
    this->_tile_len = dh;
    this->_st_ynt = st_ynt;
    this->_st_ynh = st_ynh;
    this->_st_xnt = st_xnt;
    this->_st_xnh = st_xnh;
    _copy_len = alignTileLen<T>(dh, BYTE_ALIGN);
    _half_copy_len = alignTileLen<T>(dh, BYTE_ALIGN);

    _block_idx = GetBlockIdx();

    // Init global buffer
    xGm.SetGlobalBuffer((__gm__ T *)x);
    pGm.SetGlobalBuffer((__gm__ U *)pos);
    sinGm.SetGlobalBuffer((__gm__ T *)sin);
    cosGm.SetGlobalBuffer((__gm__ T *)cos);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    // Init Queue buffer
    pipe.InitBuffer(inQue, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(outQue, BUFFER_NUM, _tile_len * sizeof(T));
    pipe.InitBuffer(sinQue, BUFFER_NUM, _half_copy_len * sizeof(T));
    pipe.InitBuffer(cosQue, BUFFER_NUM, _half_copy_len * sizeof(T));
    pipe.InitBuffer(tmpOddBuf, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpEvenBuf, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpOddBuf1, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpOddBuf2, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpEvenBuf1, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpEvenBuf2, _tile_len / 2 * sizeof(T));
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::copyIn(size_t i) {
    LocalTensor<T> inputUb = inQue.AllocTensor<T>();
    LocalTensor<T> sinUb = sinQue.AllocTensor<T>();
    LocalTensor<T> cosUb = cosQue.AllocTensor<T>();
    // Get idx of current tile in total input
    auto idx = i * _st_xnt + _block_idx * _st_xnh;
    // Copy tile current tile into UB
    DataCopy(inputUb, xGm[idx], _copy_len);
    // Copy sin cos tile
    auto pos_idx = pGm(i);
    DataCopy(sinUb, sinGm[pos_idx * _tile_len / 2], _half_copy_len);
    DataCopy(cosUb, cosGm[pos_idx * _tile_len / 2], _half_copy_len);
    // Push in operands
    inQue.EnQue(inputUb);
    sinQue.EnQue(sinUb);
    cosQue.EnQue(cosUb);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::compute(size_t i) {
    LocalTensor<T> inputUb = inQue.DeQue<T>();
    LocalTensor<T> sinUb = sinQue.DeQue<T>();
    LocalTensor<T> cosUb = cosQue.DeQue<T>();
    LocalTensor<T> outputUb = outQue.AllocTensor<T>();

    LocalTensor<T> tmpOdd = tmpOddBuf.Get<T>();
    LocalTensor<T> tmpEven = tmpEvenBuf.Get<T>();
    LocalTensor<T> tmpOdd1 = tmpOddBuf1.Get<T>();
    LocalTensor<T> tmpOdd2 = tmpOddBuf2.Get<T>();
    LocalTensor<T> tmpEven1 = tmpEvenBuf1.Get<T>();
    LocalTensor<T> tmpEven2 = tmpEvenBuf2.Get<T>();

    // separate odd and even bit elements
    uint64_t rsvdCnt = 0;
    GatherMaskParams gMaskParams = {
        1,
        static_cast<uint16_t>((_tile_len * sizeof(T) + 255) / 256), // no more than 256(<=255)
        8,
        8,
    };
    GatherMask<T>(tmpOdd, inputUb, 1, false, 0, gMaskParams, rsvdCnt);
    GatherMask<T>(tmpEven, inputUb, 2, false, 0, gMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // compute odd bit elements
    // y_odd = x_odd * cos - x_even * sin
    Mul<T>(tmpOdd1, tmpOdd, cosUb, _tile_len / 2);
    Mul<T>(tmpOdd2, tmpEven, sinUb, _tile_len / 2);
    PipeBarrier<PIPE_V>();
    Sub<T>(tmpOdd1, tmpOdd1, tmpOdd2, _tile_len / 2);

    // compute even bit elements
    // y_even = x_odd * sin + x_even * cos
    Mul<T>(tmpEven1, tmpOdd, sinUb, _tile_len / 2);
    Mul<T>(tmpEven2, tmpEven, cosUb, _tile_len / 2);
    PipeBarrier<PIPE_V>();
    Add<T>(tmpEven1, tmpEven1, tmpEven2, _tile_len / 2);

    // combine odd and even bit elements
    for (uint32_t j = 0; j < _tile_len / 2; j += 1) {
        outputUb(j * 2) = tmpOdd1(j);
        outputUb(j * 2 + 1) = tmpEven1(j);
    }

    outQue.EnQue<T>(outputUb);
    inQue.FreeTensor(inputUb);
    sinQue.FreeTensor(sinUb);
    cosQue.FreeTensor(cosUb);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::copyOut(size_t i) {
    LocalTensor<T> outputUb = outQue.DeQue<T>();
    auto idy = i * _st_ynt + _block_idx * _st_ynh;
    DataCopyExtParams params = {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm[idy], outputUb, params);
    outQue.FreeTensor(outputUb);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::process(size_t seq_len) {

    for (size_t i = 0; i < seq_len; ++i) {
        copyIn(i);
        compute(i);
        copyOut(i);
    }
}

#define ROPE_KERNEL(TYPE, POSTYPE)                                                                             \
    switch (POSTYPE) {                                                                                         \
    case INFINI_DTYPE_I8: {                                                                                    \
        RoPEKernel<TYPE, int8_t> op;                                                                           \
        op.init(y, x, pos, sin, cos, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead); \
        op.process(seq_len);                                                                                   \
        break;                                                                                                 \
    }                                                                                                          \
    case INFINI_DTYPE_I16: {                                                                                   \
        RoPEKernel<TYPE, int16_t> op;                                                                          \
        op.init(y, x, pos, sin, cos, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead); \
        op.process(seq_len);                                                                                   \
        break;                                                                                                 \
    }                                                                                                          \
    case INFINI_DTYPE_I32: {                                                                                   \
        RoPEKernel<TYPE, int32_t> op;                                                                          \
        op.init(y, x, pos, sin, cos, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead); \
        op.process(seq_len);                                                                                   \
        break;                                                                                                 \
    }                                                                                                          \
    case INFINI_DTYPE_I64: {                                                                                   \
        RoPEKernel<TYPE, int64_t> op;                                                                          \
        op.init(y, x, pos, sin, cos, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead); \
        op.process(seq_len);                                                                                   \
        break;                                                                                                 \
    }                                                                                                          \
    case INFINI_DTYPE_U8: {                                                                                    \
        RoPEKernel<TYPE, uint8_t> op;                                                                          \
        op.init(y, x, pos, sin, cos, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead); \
        op.process(seq_len);                                                                                   \
        break;                                                                                                 \
    }                                                                                                          \
    case INFINI_DTYPE_U16: {                                                                                   \
        RoPEKernel<TYPE, uint16_t> op;                                                                         \
        op.init(y, x, pos, sin, cos, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead); \
        op.process(seq_len);                                                                                   \
        break;                                                                                                 \
    }                                                                                                          \
    case INFINI_DTYPE_U32: {                                                                                   \
        RoPEKernel<TYPE, uint32_t> op;                                                                         \
        op.init(y, x, pos, sin, cos, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead); \
        op.process(seq_len);                                                                                   \
        break;                                                                                                 \
    }                                                                                                          \
    case INFINI_DTYPE_U64: {                                                                                   \
        RoPEKernel<TYPE, uint64_t> op;                                                                         \
        op.init(y, x, pos, sin, cos, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead); \
        op.process(seq_len);                                                                                   \
        break;                                                                                                 \
    }                                                                                                          \
    }

#define DEFINE_ROPE_KERNEL(KERNEL_NAME, TYPE)                         \
    __global__ __aicore__ void KERNEL_NAME(GM_ADDR y,                 \
                                           GM_ADDR x,                 \
                                           GM_ADDR pos,               \
                                           GM_ADDR sin,               \
                                           GM_ADDR cos,               \
                                           size_t seq_len,            \
                                           size_t dhead,              \
                                           ptrdiff_t y_stride_seqlen, \
                                           ptrdiff_t y_stride_nhead,  \
                                           ptrdiff_t x_stride_seqlen, \
                                           ptrdiff_t x_stride_nhead,  \
                                           int32_t pos_type) {        \
        ROPE_KERNEL(TYPE, pos_type)                                   \
    }

DEFINE_ROPE_KERNEL(rope_kernel_float, float)
DEFINE_ROPE_KERNEL(rope_kernel_half, half)

#undef DEFINE_ROPE_KERNEL

extern "C" infiniStatus_t rope_kernel_launch(
    void *y,
    void *x,
    void *pos,
    void *sin,
    void *cos,
    size_t seq_len,
    size_t nhead,
    size_t dhead,
    infiniDtype_t dtype,
    infiniDtype_t pos_type,
    ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead,
    ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead,
    void *stream) {

#define LAUNCH_ROPE_KERNEL(DTYPE_ENUM, KERNEL_NAME)                  \
    case DTYPE_ENUM:                                                 \
        KERNEL_NAME<<<nhead, nullptr, stream>>>(y, x, pos, sin, cos, \
                                                seq_len,             \
                                                dhead,               \
                                                y_stride_seqlen,     \
                                                y_stride_nhead,      \
                                                x_stride_seqlen,     \
                                                x_stride_nhead,      \
                                                pos_type);           \
        return INFINI_STATUS_SUCCESS;

    switch (dtype) {
        LAUNCH_ROPE_KERNEL(INFINI_DTYPE_F16, rope_kernel_half)
        LAUNCH_ROPE_KERNEL(INFINI_DTYPE_F32, rope_kernel_float)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
