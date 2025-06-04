#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

template <typename T>
class SwigluKernel {
public:
    __aicore__ inline SwigluKernel() {}
    __aicore__ inline void init(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                size_t batch_, size_t seq, size_t hd,
                                ptrdiff_t stride_batch_c,
                                ptrdiff_t stride_batch_a,
                                ptrdiff_t stride_batch_b,
                                ptrdiff_t stride_seq_c,
                                ptrdiff_t stride_seq_a,
                                ptrdiff_t stride_seq_b);
    __aicore__ inline void process();

private:
    __aicore__ inline void copyIn(size_t i);
    __aicore__ inline void compute(size_t i);
    __aicore__ inline void copyOut(size_t i);

private:
    GlobalTensor<T> _c_gm, _a_gm, _b_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> _in_queue_a, _in_queue_b;
    TQue<QuePosition::VECOUT, BUFFER_NUM> _out_queue_c;

    TPipe _pipe;
    float _beta_value = 1.0f;
    size_t _block_idx, _tile_len, _copy_len,
        _batch, _seq_len, _hidden_size,
        _stride_seq_a, _stride_seq_b, _stride_seq_c;
    int64_t _stride_batch_a = 1, _stride_batch_b = 1, _stride_batch_c = 1;
};

template <typename T>
__aicore__ inline void SwigluKernel<T>::init(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                             size_t batch_, size_t seq, size_t hd,
                                             ptrdiff_t stride_batch_c,
                                             ptrdiff_t stride_batch_a,
                                             ptrdiff_t stride_batch_b,
                                             ptrdiff_t stride_seq_c,
                                             ptrdiff_t stride_seq_a,
                                             ptrdiff_t stride_seq_b) {
    // Init Shape & StrideVariables
    _batch = batch_;
    _seq_len = seq;
    _hidden_size = hd;
    _stride_batch_a = stride_batch_a;
    _stride_batch_b = stride_batch_b;
    _stride_batch_c = stride_batch_c;
    _stride_seq_a = stride_seq_a;
    _stride_seq_b = stride_seq_b;
    _stride_seq_c = stride_seq_c;

    _block_idx = GetBlockIdx();
    _tile_len = _block_idx < (_hidden_size % BLOCK_NUM) ? (_hidden_size / BLOCK_NUM) + 1 : (_hidden_size / BLOCK_NUM);
    _copy_len = alignTileLen<T>(_tile_len, BYTE_ALIGN);

    // Set global tensor
    _a_gm.SetGlobalBuffer((__gm__ T *)a);
    _b_gm.SetGlobalBuffer((__gm__ T *)b);
    _c_gm.SetGlobalBuffer((__gm__ T *)c);

    // _pipe alloc memory to queue, the unit is bytes
    _pipe.InitBuffer(_in_queue_a, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_b, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_out_queue_c, BUFFER_NUM, _copy_len * sizeof(T));
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::copyIn(size_t i) {
    // Alloc tensor from queue memory
    LocalTensor<T> aLocal = _in_queue_a.AllocTensor<T>();
    LocalTensor<T> bLocal = _in_queue_b.AllocTensor<T>();
    // Get idx of current tile
    auto batch_idx = _batch == 1 ? 0 : i / _seq_len;
    auto seq_idx = _batch == 1 ? i : i % _seq_len;

    ptrdiff_t idxa = batch_idx * _stride_batch_a + seq_idx * _stride_seq_a + _block_idx * _tile_len;
    ptrdiff_t idxb = batch_idx * _stride_batch_b + seq_idx * _stride_seq_b + _block_idx * _tile_len;
    // Copy process_th tile from global tensor to local tensor
    DataCopy(aLocal, _a_gm[idxa], _copy_len);
    DataCopy(bLocal, _b_gm[idxb], _copy_len);

    // Enque input tensor to VECIN queue
    _in_queue_a.EnQue(aLocal);
    _in_queue_b.EnQue(bLocal);
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::compute(size_t i) {
    // Deque input tensors from VECIN queue
    LocalTensor<T> aLocal = _in_queue_a.DeQue<T>();
    LocalTensor<T> bLocal = _in_queue_b.DeQue<T>();
    LocalTensor<T> cLocal = _out_queue_c.AllocTensor<T>();
    // Call SwiGLU ascend api
    SwiGLU<T, false>(cLocal, aLocal, bLocal, _beta_value, _copy_len);
    // Enque result and free input
    _out_queue_c.EnQue<T>(cLocal);
    _in_queue_a.FreeTensor(aLocal);
    _in_queue_b.FreeTensor(bLocal);
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::copyOut(size_t i) {
    // Deque output tensor from VECOUT queue
    LocalTensor<T> cLocal = _out_queue_c.DeQue<T>();
    auto batch_idx = _batch == 1 ? 0 : i / _seq_len;
    auto seq_idx = _batch == 1 ? i : i % _seq_len;
    ptrdiff_t idxc = batch_idx * _stride_batch_c + seq_idx * _stride_seq_c + _block_idx * _tile_len;
    // Copy progress_th tile from local tensor to global tensor
    if (_tile_len * sizeof(T) % BYTE_ALIGN != 0) {
        DataCopyExtParams dcep = {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0};
        DataCopyPad(_c_gm[idxc], cLocal, dcep);
    } else {
        DataCopy(_c_gm[idxc], cLocal, _tile_len);
    }
    // Free output Local tensor
    _out_queue_c.FreeTensor(cLocal);
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::process() {
    for (size_t i = 0; i < _batch * _seq_len; ++i) {
        copyIn(i);
        compute(i);
        copyOut(i);
    }
}

#define DEFINE_SWIGLU_KERNEL(KERNEL_NAME, TYPE)                                 \
    __global__ __aicore__ void KERNEL_NAME(GM_ADDR c, GM_ADDR a, GM_ADDR b,     \
                                           size_t batch, size_t seq, size_t hd, \
                                           ptrdiff_t stride_batch_c,            \
                                           ptrdiff_t stride_batch_a,            \
                                           ptrdiff_t stride_batch_b,            \
                                           ptrdiff_t stride_seq_c,              \
                                           ptrdiff_t stride_seq_a,              \
                                           ptrdiff_t stride_seq_b) {            \
        SwigluKernel<TYPE> op;                                                  \
        op.init(c, a, b,                                                        \
                batch, seq, hd,                                                 \
                stride_batch_c, stride_batch_a, stride_batch_b,                 \
                stride_seq_c, stride_seq_a, stride_seq_b);                      \
        op.process();                                                           \
    }

DEFINE_SWIGLU_KERNEL(swiglu_kernel_half, half)
DEFINE_SWIGLU_KERNEL(swiglu_kernel_float, float)

#undef DEFINE_SWIGLU_KERNEL

extern "C" infiniStatus_t swiglu_kernel_launch(
    void *c, void *a, void *b,
    infiniDtype_t dtype, size_t batch, size_t seq, size_t hd,
    ptrdiff_t stride_batch_c, ptrdiff_t stride_batch_a, ptrdiff_t stride_batch_b,
    ptrdiff_t stride_seq_c, ptrdiff_t stride_seq_a, ptrdiff_t stride_seq_b, void *stream) {

#define LAUNCH_SWIGLU_KERNEL(DTYPE_ENUM, KERNEL_NAME)       \
    case DTYPE_ENUM:                                        \
        KERNEL_NAME<<<BLOCK_NUM, nullptr, stream>>>(        \
            c, a, b,                                        \
            batch,                                          \
            seq,                                            \
            hd,                                             \
            stride_batch_c, stride_batch_a, stride_batch_b, \
            stride_seq_c, stride_seq_a, stride_seq_b);      \
        return INFINI_STATUS_SUCCESS;

    switch (dtype) {
        LAUNCH_SWIGLU_KERNEL(INFINI_DTYPE_F16, swiglu_kernel_half)
        LAUNCH_SWIGLU_KERNEL(INFINI_DTYPE_F32, swiglu_kernel_float)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_SWIGLU_KERNEL
}
