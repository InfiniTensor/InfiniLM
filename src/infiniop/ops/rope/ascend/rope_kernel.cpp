#include "../../../../../include/infinicore.h"
#include "kernel_operator.h"

constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_ALIGN = 32;

using namespace AscendC;

template <typename T>
class RoPEKernel {
public:
    __aicore__ inline RoPEKernel() {}
    // Init op
    // pos position vector
    // x input tensor
    // y output tensor
    // tensor shape [nt, nh, dh]
    // make block_num = nh, tile_len = dh
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR x, GM_ADDR pos, GM_ADDR sin, GM_ADDR cos,
                                int32_t dh,
                                int32_t st_ynt, int32_t st_ynh,
                                int32_t st_xnt, int32_t st_xnh);
    __aicore__ inline void Process(int32_t seq_len);

private:
    // Copy a tile into UB
    __aicore__ inline void CopyIn(int32_t i);
    __aicore__ inline void Compute(int32_t i);
    __aicore__ inline void CopyOut(int32_t i);

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
    GlobalTensor<uint32_t> pGm;
    GlobalTensor<T> sinGm;
    GlobalTensor<T> cosGm;

    uint32_t _block_idx;
    uint32_t _tile_len;
    uint32_t _copy_len;
    uint32_t _half_copy_len;

    // stridey[st_ynt_, st_ynh_, 1]
    int32_t st_ynt_;
    int32_t st_ynh_;
    // stridex[st_xnt_, st_xnh_, 1]
    int32_t st_xnt_;
    int32_t st_xnh_;
};

template <typename T>
__aicore__ inline void RoPEKernel<T>::Init(GM_ADDR y, GM_ADDR x, GM_ADDR pos, GM_ADDR sin, GM_ADDR cos,
                                           int32_t dh,
                                           int32_t st_ynt, int32_t st_ynh,
                                           int32_t st_xnt, int32_t st_xnh) {
    this->_tile_len = dh;
    this->st_ynt_ = st_ynt;
    this->st_ynh_ = st_ynh;
    this->st_xnt_ = st_xnt;
    this->st_xnh_ = st_xnh;
    _copy_len = (_tile_len * sizeof(T)) % BYTE_ALIGN == 0 ? _tile_len : (_tile_len * sizeof(T) + (BYTE_ALIGN - _tile_len * sizeof(T) % BYTE_ALIGN)) / sizeof(T);
    _half_copy_len = (_tile_len / 2 * sizeof(T)) % BYTE_ALIGN == 0 ? _tile_len / 2 : (_tile_len / 2 * sizeof(T) + (BYTE_ALIGN - _tile_len / 2 * sizeof(T) % BYTE_ALIGN)) / sizeof(T);

    _block_idx = GetBlockIdx();

    // Init global buffer
    xGm.SetGlobalBuffer((__gm__ T *)x);
    pGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(pos));
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

template <typename T>
__aicore__ inline void RoPEKernel<T>::CopyIn(int32_t i) {
    LocalTensor<T> inputUb = inQue.AllocTensor<T>();
    LocalTensor<T> sinUb = sinQue.AllocTensor<T>();
    LocalTensor<T> cosUb = cosQue.AllocTensor<T>();
    // Get idx of current tile in total input
    auto idx = i * st_xnt_ + _block_idx * st_xnh_;
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

template <typename T>
__aicore__ inline void RoPEKernel<T>::Compute(int32_t i) {
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

template <typename T>
__aicore__ inline void RoPEKernel<T>::CopyOut(int32_t i) {
    LocalTensor<T> outputUb = outQue.DeQue<T>();
    auto idy = i * st_ynt_ + _block_idx * st_ynh_;
    DataCopyExtParams params = {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm[idy], outputUb, params);
    outQue.FreeTensor(outputUb);
}

template <typename T>
__aicore__ inline void RoPEKernel<T>::Process(int32_t nt) {

    for (int32_t i = 0; i < nt; ++i) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

__global__ __aicore__ void rope_f16_kernel(GM_ADDR y, GM_ADDR x, GM_ADDR pos, GM_ADDR sin, GM_ADDR cos,
                                           int32_t seq_len, int32_t dhead,
                                           int32_t y_stride_seqlen, int32_t y_stride_nhead,
                                           int32_t x_stride_seqlen, int32_t x_stride_nhead) {
    RoPEKernel<half> op;
    op.Init(y, x, pos, sin, cos, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead);
    op.Process(seq_len);
}

__global__ __aicore__ void rope_f32_kernel(GM_ADDR y, GM_ADDR x, GM_ADDR pos, GM_ADDR sin, GM_ADDR cos,
                                           int32_t seq_len, int32_t dhead,
                                           int32_t y_stride_seqlen, int32_t y_stride_nhead,
                                           int32_t x_stride_seqlen, int32_t x_stride_nhead) {
    RoPEKernel<float> op;
    op.Init(y, x, pos, sin, cos, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead);
    op.Process(seq_len);
}

extern "C" infiniStatus_t rope_kernel_launch(void *y,
                                             void *x,
                                             void *pos,
                                             void *sin,
                                             void *cos,
                                             int32_t seq_len,
                                             int32_t nhead,
                                             int32_t dhead,
                                             int32_t data_type,
                                             int32_t y_stride_seqlen,
                                             int32_t y_stride_nhead,
                                             int32_t x_stride_seqlen,
                                             int32_t x_stride_nhead,
                                             void *stream) {
    switch (data_type) {
    case 12: // float16
        rope_f16_kernel<<<nhead, nullptr, stream>>>(y, x, pos, sin, cos, seq_len, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead);
        break;
    case 13: // float32
        rope_f32_kernel<<<nhead, nullptr, stream>>>(y, x, pos, sin, cos, seq_len, dhead, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
