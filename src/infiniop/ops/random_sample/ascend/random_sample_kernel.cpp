#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

template <typename T>
class RandomSampleKernel {
public:
    __aicore__ inline RandomSampleKernel() {}
    __aicore__ inline void init(GM_ADDR probs, GM_ADDR result, GM_ADDR topk_val_addr, GM_ADDR topk_idx_addr, float random_val, float topp, int topk, float temperature, int32_t n);
    __aicore__ inline void process();

private:
    __aicore__ inline void copyIn();
    __aicore__ inline void copyOut();
    __aicore__ inline void compute();
    __aicore__ inline void SoftMax(LocalTensor<T> &topkValIn,
                                   LocalTensor<T> &softMaxOut);
    __aicore__ inline void InclusiveSum(LocalTensor<T> &topkValIn,
                                        LocalTensor<T> &topkValOut);
    __aicore__ inline void RandomSample(LocalTensor<T> &valIn,
                                        LocalTensor<int64_t> &Index,
                                        LocalTensor<int64_t> &result);

    GlobalTensor<T> _pGM,
        _topk_valGM;
    GlobalTensor<int64_t> _topk_idxGM, _resGM;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> _topk_valQue;
    TQue<QuePosition::VECIN, 1> _topk_idxQue;
    TQue<QuePosition::VECOUT, 1> _resQue;

    TBuf<TPosition::VECCALC> _inBuf;
    TBuf<TPosition::VECCALC> _tmp1Buf;
    TBuf<TPosition::VECCALC> _tmp2Buf;
    TBuf<TPosition::VECCALC> _tmp3Buf;
    TBuf<TPosition::VECCALC> _softmax_OutBuf;
    TBuf<TPosition::VECCALC> _inclusive_sum_OutBuf;

    int32_t _topk;
    int32_t _voc;
    float _random_val;
    float _topp;
    float _invTemp;
    float _negMax = 0.f;
    float _sum = 0.f;

    int32_t _topkAligned;
    int32_t _topkIdxAligned;
    int32_t _vocAligned;
    int32_t _bufferLen;
};

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::init(GM_ADDR probs, GM_ADDR result, GM_ADDR topk_val_addr, GM_ADDR topk_idx_addr, float random_val, float topp, int topk, float temperature, int32_t n) {
    _topk = topk;
    _voc = n;
    _random_val = random_val;
    _topp = topp;
    _invTemp = 1.0f / temperature;

    // CumSumInfo
    _topkAligned = alignTileLen<T>(_topk, BYTE_ALIGN);
    _vocAligned = alignTileLen<T>(_voc, BYTE_ALIGN);
    _topkIdxAligned = (_topk + 3) / 4 * 4;
    _bufferLen = _topkAligned > BLOCK_LEN ? _topkAligned : BLOCK_LEN;

    // Set GlobalTensor
    _pGM.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(probs), _voc);
    _topk_valGM.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(topk_val_addr), _topk);
    _topk_idxGM.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(topk_idx_addr), _topk);
    _resGM.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(result), 1);

    // Global input and output
    pipe.InitBuffer(_topk_valQue, 1, _topkAligned * sizeof(T));
    pipe.InitBuffer(_topk_idxQue, 1, _topkIdxAligned * sizeof(int64_t));
    pipe.InitBuffer(_resQue, 1, BYTE_ALIGN); // 32 bytes for aligned
    pipe.InitBuffer(_inBuf, BLOCK_LEN * sizeof(T));
    pipe.InitBuffer(_tmp1Buf, _bufferLen * sizeof(T));
    pipe.InitBuffer(_tmp2Buf, _bufferLen * sizeof(T));
    pipe.InitBuffer(_tmp3Buf, _bufferLen * sizeof(T));
    pipe.InitBuffer(_softmax_OutBuf, _topkAligned * sizeof(T));
    pipe.InitBuffer(_inclusive_sum_OutBuf, _topkAligned * sizeof(T));
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::process() {
    copyIn();
    compute();
    copyOut();
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::SoftMax(LocalTensor<T> &topkValIn,
                                                      LocalTensor<T> &softMaxOut) {
    float invSum = 1.0f / _sum;
    LocalTensor<T> tmpBuffer = _tmp1Buf.Get<T>();
    LocalTensor<T> tmpBuffer2 = _tmp2Buf.Get<T>();
    LocalTensor<T> tmpBuffer3 = _tmp3Buf.Get<T>();
    Adds(tmpBuffer, topkValIn, static_cast<T>(_negMax), _topk);
    Muls(tmpBuffer2, tmpBuffer, static_cast<T>(_invTemp), _topk);
    Exp(tmpBuffer3, tmpBuffer2, _topk);
    Muls(softMaxOut, tmpBuffer3, static_cast<T>(invSum), _topk);
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::InclusiveSum(LocalTensor<T> &topkValIn,
                                                           LocalTensor<T> &topkValOut) {
    static constexpr CumSumConfig cumSumConfig{true, false, false};
    LocalTensor<T> lastRowLocal;
    CumSum<T, cumSumConfig>(topkValOut, lastRowLocal, topkValIn,
                            {1, static_cast<uint32_t>(_topkAligned)});
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::RandomSample(LocalTensor<T> &valIn,
                                                           LocalTensor<int64_t> &Index,
                                                           LocalTensor<int64_t> &result) {
    int end = 0;
    for (end = 0; end < _topk; end++) {
        if (static_cast<float>(valIn(end)) >= _topp) {
            break;
        }
    }
    if (end < _topk - 1) {
        end += 1;
    } else {
        end = _topk;
    }

    auto random_val = _random_val * static_cast<float>(valIn(end - 1));
    for (int i = 0; i < end; i++) {
        if (random_val < static_cast<float>(valIn(i))) {
            result(0) = Index(i);
            return;
        }
    }
    result(0) = Index(end - 1);
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::copyIn() {
    LocalTensor<T> topkValLocal = _topk_valQue.AllocTensor<T>();
    LocalTensor<int64_t> topkIdxLocal = _topk_idxQue.AllocTensor<int64_t>();
    DataCopy(topkValLocal, _topk_valGM, _topkAligned);
    DataCopy(topkIdxLocal, _topk_idxGM, _topkIdxAligned);
    // Get Max val of input
    _negMax = -static_cast<float>(topkValLocal(0));

    // Copy in p and compute _sum
    int32_t repeatTimes = _voc / BLOCK_LEN;
    int32_t remainder = _voc % BLOCK_LEN;
    float sum_s = 0.f;
    LocalTensor<T> _inBuffer = _inBuf.Get<T>();
    LocalTensor<T> tmpBuffer = _tmp1Buf.Get<T>();
    LocalTensor<T> tmpBuffer2 = _tmp2Buf.Get<T>();
    LocalTensor<T> tmpBuffer3 = _tmp3Buf.Get<T>();
    for (int32_t i = 0; i < repeatTimes; i++) {
        DataCopy(_inBuffer, _pGM[i * BLOCK_LEN], BLOCK_LEN);
        Adds(tmpBuffer, _inBuffer, static_cast<T>(_negMax), BLOCK_LEN);
        Muls(tmpBuffer2, tmpBuffer, static_cast<T>(_invTemp), BLOCK_LEN);
        Exp(tmpBuffer3, tmpBuffer2, BLOCK_LEN);
        sum_s = 0.f;
        for (int j = 0; j < BLOCK_LEN; ++j) {
            sum_s += static_cast<float>(tmpBuffer3(j));
        }
        _sum += sum_s;
    }
    if (remainder != 0) {
        int32_t remainderAligned = alignTileLen<T>(remainder, BYTE_ALIGN);
        DataCopy(_inBuffer, _pGM[repeatTimes * BLOCK_LEN], remainderAligned);
        Adds(tmpBuffer, _inBuffer, static_cast<T>(_negMax), remainder);
        Muls(tmpBuffer2, tmpBuffer, static_cast<T>(_invTemp), remainder);
        Exp(tmpBuffer3, tmpBuffer2, remainder);
        sum_s = 0.f;
        for (int i = 0; i < remainder; ++i) {
            sum_s += static_cast<float>(tmpBuffer3(i));
        }
        _sum += sum_s;
    }

    _topk_valQue.EnQue(topkValLocal);
    _topk_idxQue.EnQue(topkIdxLocal);
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::compute() {
    // Get input data
    LocalTensor<T> topkValLocal = _topk_valQue.DeQue<T>();

    // SoftMax
    LocalTensor<T> softMaxOutLocal = _softmax_OutBuf.Get<T>();
    SoftMax(topkValLocal, softMaxOutLocal);

    // InclusiveSum
    LocalTensor<T> inclusiveOutLocal = _inclusive_sum_OutBuf.Get<T>();
    InclusiveSum(softMaxOutLocal, inclusiveOutLocal);

    // randomSample
    LocalTensor<int64_t> topkIdxLocal = _topk_idxQue.DeQue<int64_t>();
    LocalTensor<int64_t> resultLocal = _resQue.AllocTensor<int64_t>();
    RandomSample(inclusiveOutLocal, topkIdxLocal, resultLocal);

    _topk_valQue.FreeTensor(topkValLocal);
    _topk_idxQue.FreeTensor(topkIdxLocal);
    _resQue.EnQue(resultLocal);
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::copyOut() {
    LocalTensor<int64_t> resLocal = _resQue.DeQue<int64_t>();
    DataCopy(_resGM, resLocal, BYTE_ALIGN / sizeof(int64_t));
    _resQue.FreeTensor(resLocal);
}

extern "C" __global__ __aicore__ void random_sample_kernel_fp16(
    GM_ADDR probs,
    GM_ADDR result,
    GM_ADDR topk_val_addr,
    GM_ADDR topk_idx_addr,
    float random_val,
    float topp,
    int topk,
    float temperature,
    int32_t n) {
    RandomSampleKernel<half> op;
    op.init(probs, result, topk_val_addr, topk_idx_addr, random_val, topp, topk, temperature, n);
    op.process();
}

extern "C" __global__ __aicore__ void random_sample_kernel_fp32(
    GM_ADDR probs,
    GM_ADDR result,
    GM_ADDR topk_val_addr,
    GM_ADDR topk_idx_addr,
    float random_val,
    float topp,
    int topk,
    float temperature,
    int32_t n) {
    RandomSampleKernel<float> op;
    op.init(probs, result, topk_val_addr, topk_idx_addr, random_val, topp, topk, temperature, n);
    op.process();
}

extern "C" infiniStatus_t random_sample_kernel_launch(
    void *probs,
    void *result,
    void *topk_val_addr,
    void *topk_idx_addr,
    float random_val,
    float topp,
    int topk,
    float temperature,
    uint64_t n,
    infiniDtype_t dt_p,
    void *stream) {
    switch (dt_p) {
    case INFINI_DTYPE_F16:
        random_sample_kernel_fp16<<<1, nullptr, stream>>>(probs, result, topk_val_addr, topk_idx_addr, random_val, topp, topk, temperature, n);
        break;
    case INFINI_DTYPE_F32:
        random_sample_kernel_fp32<<<1, nullptr, stream>>>(probs, result, topk_val_addr, topk_idx_addr, random_val, topp, topk, temperature, n);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
