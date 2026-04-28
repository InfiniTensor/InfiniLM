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
    TQue<QuePosition::VECIN, BUFFER_NUM> _in_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> _sin_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> _cos_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> _out_que;
    TBuf<TPosition::VECCALC> _tmp_odd_buf;
    TBuf<TPosition::VECCALC> _tmp_even_buf;
    TBuf<TPosition::VECCALC> _tmp_odd_buf1;
    TBuf<TPosition::VECCALC> _tmp_odd_buf2;
    TBuf<TPosition::VECCALC> _tmp_even_buf1;
    TBuf<TPosition::VECCALC> _tmp_even_buf2;

    GlobalTensor<T> _x_gm, _y_gm;
    GlobalTensor<U> _p_gm;
    GlobalTensor<T> _sin_gm;
    GlobalTensor<T> _cos_gm;

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
    _x_gm.SetGlobalBuffer((__gm__ T *)x);
    _p_gm.SetGlobalBuffer((__gm__ U *)pos);
    _sin_gm.SetGlobalBuffer((__gm__ T *)sin);
    _cos_gm.SetGlobalBuffer((__gm__ T *)cos);
    _y_gm.SetGlobalBuffer((__gm__ T *)y);

    // Init Queue buffer
    pipe.InitBuffer(_in_que, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(_out_que, BUFFER_NUM, _tile_len * sizeof(T));
    pipe.InitBuffer(_sin_que, BUFFER_NUM, _half_copy_len * sizeof(T));
    pipe.InitBuffer(_cos_que, BUFFER_NUM, _half_copy_len * sizeof(T));
    pipe.InitBuffer(_tmp_odd_buf, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(_tmp_even_buf, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(_tmp_odd_buf1, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(_tmp_odd_buf2, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(_tmp_even_buf1, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(_tmp_even_buf2, _tile_len / 2 * sizeof(T));
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::copyIn(size_t i) {
    LocalTensor<T> input_ub = _in_que.AllocTensor<T>();
    LocalTensor<T> sin_ub = _sin_que.AllocTensor<T>();
    LocalTensor<T> cos_ub = _cos_que.AllocTensor<T>();
    // Get idx of current tile in total input
    auto idx = i * _st_xnt + _block_idx * _st_xnh;
    // Copy tile current tile into UB
    DataCopy(input_ub, _x_gm[idx], _copy_len);
    // Copy sin cos tile
    auto pos_idx = _p_gm(i);
    DataCopy(sin_ub, _sin_gm[pos_idx * _tile_len / 2], _half_copy_len);
    DataCopy(cos_ub, _cos_gm[pos_idx * _tile_len / 2], _half_copy_len);
    // Push in operands
    _in_que.EnQue(input_ub);
    _sin_que.EnQue(sin_ub);
    _cos_que.EnQue(cos_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::compute(size_t i) {
    LocalTensor<T> input_ub = _in_que.DeQue<T>();
    LocalTensor<T> sin_ub = _sin_que.DeQue<T>();
    LocalTensor<T> cos_ub = _cos_que.DeQue<T>();
    LocalTensor<T> output_ub = _out_que.AllocTensor<T>();

    LocalTensor<T> tmp_odd = _tmp_odd_buf.Get<T>();
    LocalTensor<T> tmp_even = _tmp_even_buf.Get<T>();
    LocalTensor<T> tmp_odd1 = _tmp_odd_buf1.Get<T>();
    LocalTensor<T> tmp_odd2 = _tmp_odd_buf2.Get<T>();
    LocalTensor<T> tmp_even1 = _tmp_even_buf1.Get<T>();
    LocalTensor<T> tmp_even2 = _tmp_even_buf2.Get<T>();

    // separate odd and even bit elements
    uint64_t rsvdCnt = 0;
    GatherMaskParams gMaskParams = {
        1,
        static_cast<uint16_t>((_tile_len * sizeof(T) + 255) / 256), // no more than 256(<=255)
        8,
        8,
    };
    GatherMask<T>(tmp_odd, input_ub, 1, false, 0, gMaskParams, rsvdCnt);
    GatherMask<T>(tmp_even, input_ub, 2, false, 0, gMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // compute odd bit elements
    // y_odd = x_odd * cos - x_even * sin
    Mul<T>(tmp_odd1, tmp_odd, cos_ub, _tile_len / 2);
    Mul<T>(tmp_odd2, tmp_even, sin_ub, _tile_len / 2);
    PipeBarrier<PIPE_V>();
    Sub<T>(tmp_odd1, tmp_odd1, tmp_odd2, _tile_len / 2);

    // compute even bit elements
    // y_even = x_odd * sin + x_even * cos
    Mul<T>(tmp_even1, tmp_odd, sin_ub, _tile_len / 2);
    Mul<T>(tmp_even2, tmp_even, cos_ub, _tile_len / 2);
    PipeBarrier<PIPE_V>();
    Add<T>(tmp_even1, tmp_even1, tmp_even2, _tile_len / 2);

    // combine odd and even bit elements
    for (uint32_t j = 0; j < _tile_len / 2; j += 1) {
        output_ub(j * 2) = tmp_odd1(j);
        output_ub(j * 2 + 1) = tmp_even1(j);
    }

    _out_que.EnQue<T>(output_ub);
    _in_que.FreeTensor(input_ub);
    _sin_que.FreeTensor(sin_ub);
    _cos_que.FreeTensor(cos_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::copyOut(size_t i) {
    LocalTensor<T> output_ub = _out_que.DeQue<T>();
    auto idy = i * _st_ynt + _block_idx * _st_ynh;
    DataCopyExtParams params = {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0};
    DataCopyPad(_y_gm[idy], output_ub, params);
    _out_que.FreeTensor(output_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::process(size_t seq_len) {

    for (size_t i = 0; i < seq_len; ++i) {
        copyIn(i);
        compute(i);
        copyOut(i);
    }
}

#define ROPE_KERNEL_INIT_ARGS y, x, pos, sin, cos, dhead,      \
                              y_stride_seqlen, y_stride_nhead, \
                              x_stride_seqlen, x_stride_nhead

#define CASE_POSTYPE(POS_TYPE_ENUM, TYPE, POS_T) \
    case POS_TYPE_ENUM: {                        \
        RoPEKernel<TYPE, POS_T> op;              \
        op.init(ROPE_KERNEL_INIT_ARGS);          \
        op.process(seq_len);                     \
        break;                                   \
    }

#define ROPE_KERNEL(TYPE, POSTYPE)                     \
    switch (POSTYPE) {                                 \
        CASE_POSTYPE(INFINI_DTYPE_I8, TYPE, int8_t)    \
        CASE_POSTYPE(INFINI_DTYPE_I16, TYPE, int16_t)  \
        CASE_POSTYPE(INFINI_DTYPE_I32, TYPE, int32_t)  \
        CASE_POSTYPE(INFINI_DTYPE_I64, TYPE, int64_t)  \
        CASE_POSTYPE(INFINI_DTYPE_U8, TYPE, uint8_t)   \
        CASE_POSTYPE(INFINI_DTYPE_U16, TYPE, uint16_t) \
        CASE_POSTYPE(INFINI_DTYPE_U32, TYPE, uint32_t) \
        CASE_POSTYPE(INFINI_DTYPE_U64, TYPE, uint64_t) \
    default:                                           \
        break;                                         \
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
#undef ROPE_KERNEL
#undef CASE_POSTYPE
#undef ROPE_KERNEL_INIT_ARGS

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
