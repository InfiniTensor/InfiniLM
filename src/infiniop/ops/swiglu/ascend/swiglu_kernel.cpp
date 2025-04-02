#include "../../../../../include/infinicore.h"
#include "kernel_operator.h"

constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_ALIGN = 32;
// ubsize = 196KB

using namespace AscendC;

template <typename T>
class SwigluKernel {
public:
    __aicore__ inline SwigluKernel() {}
    __aicore__ inline void Init(GM_ADDR c, GM_ADDR a, GM_ADDR b, int batch_, int seq, int hd,
                                int stride_batch_c, int stride_batch_a, int stride_batch_b,
                                int stride_seq_c, int stride_seq_a, int stride_seq_b);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t i);
    __aicore__ inline void Compute(int32_t i);
    __aicore__ inline void CopyOut(int32_t i);

private:
    GlobalTensor<T> cGm, aGm, bGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueC;

    TPipe pipe;
    uint32_t _data_size = 0;
    float _beta_value = 1.0f;
    uint32_t _block_idx, _tile_len, _copy_len;
    uint32_t batch, seq_len, hidden_size;
    int32_t strideBatchA = 1, strideBatchB = 1, strideBatchC = 1;
    int32_t strideSeqA, strideSeqB, strideSeqC;
};

template <typename T>
__aicore__ inline void SwigluKernel<T>::Init(GM_ADDR c, GM_ADDR a, GM_ADDR b, int batch_, int seq, int hd,
                                             int stride_batch_c, int stride_batch_a, int stride_batch_b,
                                             int stride_seq_c, int stride_seq_a, int stride_seq_b) {
    // Init Shape & StrideVariables
    batch = batch_;
    seq_len = seq;
    hidden_size = hd;
    strideBatchA = stride_batch_a;
    strideBatchB = stride_batch_b;
    strideBatchC = stride_batch_c;
    strideSeqA = stride_seq_a;
    strideSeqB = stride_seq_b;
    strideSeqC = stride_seq_c;

    _block_idx = GetBlockIdx();
    _tile_len = _block_idx < (hidden_size % BLOCK_NUM) ? (hidden_size / BLOCK_NUM) + 1 : (hidden_size / BLOCK_NUM);
    _copy_len = (_tile_len * sizeof(T)) % BYTE_ALIGN == 0 ? _tile_len : (_tile_len * sizeof(T) + (BYTE_ALIGN - _tile_len * sizeof(T) % BYTE_ALIGN)) / sizeof(T);

    // Set global tensor
    aGm.SetGlobalBuffer((__gm__ T *)a);
    bGm.SetGlobalBuffer((__gm__ T *)b);
    cGm.SetGlobalBuffer((__gm__ T *)c);

    // Pipe alloc memory to queue, the unit is bytes
    pipe.InitBuffer(inQueueA, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(inQueueB, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(outQueueC, BUFFER_NUM, _copy_len * sizeof(T));
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::CopyIn(int32_t i) {
    // Alloc tensor from queue memory
    LocalTensor<T> aLocal = inQueueA.AllocTensor<T>();
    LocalTensor<T> bLocal = inQueueB.AllocTensor<T>();
    // Get idx of current tile
    auto batchIdx = batch == 1 ? 0 : i / seq_len;
    auto seqIdx = batch == 1 ? i : i % seq_len;

    int32_t idxa = batchIdx * strideBatchA + seqIdx * strideSeqA + _block_idx * _tile_len;
    int32_t idxb = batchIdx * strideBatchB + seqIdx * strideSeqB + _block_idx * _tile_len;
    // Copy process_th tile from global tensor to local tensor
    DataCopy(aLocal, aGm[idxa], _copy_len);
    DataCopy(bLocal, bGm[idxb], _copy_len);

    // Enque input tensor to VECIN queue
    inQueueA.EnQue(aLocal);
    inQueueB.EnQue(bLocal);
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::Compute(int32_t i) {
    // Deque input tensors from VECIN queue
    LocalTensor<T> aLocal = inQueueA.DeQue<T>();
    LocalTensor<T> bLocal = inQueueB.DeQue<T>();
    LocalTensor<T> cLocal = outQueueC.AllocTensor<T>();
    // Call SwiGLU ascend api
    SwiGLU<T, false>(cLocal, aLocal, bLocal, _beta_value, _copy_len);
    // Enque result and free input
    outQueueC.EnQue<T>(cLocal);
    inQueueA.FreeTensor(aLocal);
    inQueueB.FreeTensor(bLocal);
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::CopyOut(int32_t i) {
    // Deque output tensor from VECOUT queue
    LocalTensor<T> cLocal = outQueueC.DeQue<T>();
    auto batchIdx = batch == 1 ? 0 : i / seq_len;
    auto seqIdx = batch == 1 ? i : i % seq_len;
    int32_t idxc = batchIdx * strideBatchC + seqIdx * strideSeqC + _block_idx * _tile_len;
    // Copy progress_th tile from local tensor to global tensor
    if (_tile_len * sizeof(T) % BYTE_ALIGN != 0) {
        DataCopyExtParams dcep = {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0};
        DataCopyPad(cGm[idxc], cLocal, dcep);
    } else {
        DataCopy(cGm[idxc], cLocal, _tile_len);
    }
    // Free output Local tensor
    outQueueC.FreeTensor(cLocal);
}

template <typename T>
__aicore__ inline void SwigluKernel<T>::Process() {
    for (int32_t i = 0; i < batch * seq_len; ++i) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

__global__ __aicore__ void swiglu_kernel_half(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                              int batch, int seq, int hd,
                                              int stride_batch_c, int stride_batch_a, int stride_batch_b,
                                              int stride_seq_c, int stride_seq_a, int stride_seq_b) {
    SwigluKernel<half> op;
    op.Init(c, a, b, batch, seq, hd, stride_batch_c, stride_batch_a, stride_batch_b, stride_seq_c, stride_seq_a, stride_seq_b);
    op.Process();
}

__global__ __aicore__ void swiglu_kernel_float(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                               int batch, int seq, int hd,
                                               int stride_batch_c, int stride_batch_a, int stride_batch_b,
                                               int stride_seq_c, int stride_seq_a, int stride_seq_b) {
    SwigluKernel<float> op;
    op.Init(c, a, b, batch, seq, hd, stride_batch_c, stride_batch_a, stride_batch_b, stride_seq_c, stride_seq_a, stride_seq_b);
    op.Process();
}

extern "C" infiniStatus_t swiglu_kernel_launch(
    void *c, void *a, void *b,
    int dtype, int batch, int seq, int hd,
    int stride_batch_c, int stride_batch_a, int stride_batch_b,
    int stride_seq_c, int stride_seq_a, int stride_seq_b, void *stream) {

    switch (dtype) {
    case 12:
        swiglu_kernel_half<<<BLOCK_NUM, nullptr, stream>>>(c, a, b, batch, seq, hd, stride_batch_c, stride_batch_a, stride_batch_b, stride_seq_c, stride_seq_a, stride_seq_b);
        return INFINI_STATUS_SUCCESS;
    case 13:
        swiglu_kernel_float<<<BLOCK_NUM, nullptr, stream>>>(c, a, b, batch, seq, hd, stride_batch_c, stride_batch_a, stride_batch_b, stride_seq_c, stride_seq_a, stride_seq_b);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
