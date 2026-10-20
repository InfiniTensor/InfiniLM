#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelRmsNorm {
 public:
  __aicore__ inline KernelRmsNorm() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                              int64_t totalRows, int64_t dimLength,
                              int64_t dimLengthAlign, int64_t formerNum,
                              int64_t formerLength, int64_t tailLength,
                              float eps) {
    this->dimLength = dimLength;
    this->dimLengthAlign = dimLengthAlign;
    this->eps = eps;

    // Block-level tiling: determine row range for this core.
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t rowOffset;

    if (blockIdx < formerNum) {
      this->blockRows = formerLength;
      rowOffset = formerLength * blockIdx;
    } else {
      this->blockRows = tailLength;
      int64_t tailIdx = blockIdx - formerNum;
      rowOffset = formerLength * formerNum + tailLength * tailIdx;
    }

    // Global memory pointers.
    xGm.SetGlobalBuffer((__gm__ T*)x + rowOffset * dimLengthAlign,
                        this->blockRows * dimLengthAlign);
    yGm.SetGlobalBuffer((__gm__ T*)y + rowOffset * dimLengthAlign,
                        this->blockRows * dimLengthAlign);
    weightGm.SetGlobalBuffer((__gm__ float*)weight, dimLengthAlign);

    int32_t dimLenAlign = static_cast<int32_t>(this->dimLengthAlign);

    // I/O queues (double-buffered).
    pipe.InitBuffer(inQueueX, BUFFER_NUM,
                    dimLenAlign * static_cast<int32_t>(sizeof(T)));
    pipe.InitBuffer(outQueueY, BUFFER_NUM,
                    dimLenAlign * static_cast<int32_t>(sizeof(T)));

    // Weight buffer (fp32, loaded once, reused for all rows).
    pipe.InitBuffer(weightBuf,
                    dimLenAlign * static_cast<int32_t>(sizeof(float)));

    // FP16 path needs extra fp32 compute buffers.
    if constexpr (sizeof(T) == 2) {
      pipe.InitBuffer(xFp32Buf,
                      dimLenAlign * static_cast<int32_t>(sizeof(float)));
      pipe.InitBuffer(tmpFp32Buf,
                      dimLenAlign * static_cast<int32_t>(sizeof(float)));
    }

    // ReduceSum temporary buffer (size per API formula).
    constexpr int32_t ELEMS_PER_REPEAT = 256 / sizeof(float);
    constexpr int32_t ELEMS_PER_BLOCK = 32 / sizeof(float);
    int32_t firstMaxRepeat =
        (dimLenAlign + ELEMS_PER_REPEAT - 1) / ELEMS_PER_REPEAT;
    int32_t reduceTmpSize =
        ((firstMaxRepeat + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK) *
        ELEMS_PER_BLOCK;
    pipe.InitBuffer(reduceTmpBuf,
                    reduceTmpSize * static_cast<int32_t>(sizeof(float)));

    // Scalar buffer for reduction result (8 floats = 32 bytes).
    pipe.InitBuffer(sumBuf, 32);

    // Load weight (fp32) from GM into `weightBuf`.
    AscendC::LocalTensor<float> wLocal = weightBuf.Get<float>();
    AscendC::DataCopyExtParams wParams{
        1, static_cast<uint32_t>(dimLenAlign * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> wPad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(wLocal, weightGm, wParams, wPad);

    // Ensure weight DMA completes before compute.
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void Process() {
    for (int64_t row = 0; row < this->blockRows; ++row) {
      CopyIn(row);
      Compute(row);
      CopyOut(row);
    }
  }

 private:
  __aicore__ inline void CopyIn(int64_t row) {
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(this->dimLengthAlign * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(xLocal, xGm[row * this->dimLengthAlign], params, pad);
    inQueueX.EnQue(xLocal);
  }

  __aicore__ inline void Compute(int64_t row) {
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

    AscendC::LocalTensor<float> wLocal = weightBuf.Get<float>();
    AscendC::LocalTensor<float> rTmp = reduceTmpBuf.Get<float>();
    AscendC::LocalTensor<float> sLocal = sumBuf.Get<float>();

    int32_t dimLen = static_cast<int32_t>(this->dimLength);
    int32_t dimLenAlign = static_cast<int32_t>(this->dimLengthAlign);

    if constexpr (sizeof(T) == 4) {
      // ---- FP32 path: compute directly. ----

      // Step 1: x^2 into yLocal (reuse output buffer temporarily).
      AscendC::Mul(yLocal, xLocal, xLocal, dimLenAlign);

      // Step 2: ReduceSum(x^2) -> sLocal[0].
      // ReduceSum may modify src (yLocal), but we overwrite it later.
      AscendC::ReduceSum(sLocal, yLocal, rTmp, dimLenAlign);

      // Step 3-5: scale = 1 / sqrt(mean(x^2) + eps).
      float sumVal = sLocal.GetValue(0);
      float meanVal = sumVal / static_cast<float>(dimLen) + this->eps;
      sLocal.SetValue(0, meanVal);
      AscendC::Sqrt(sLocal, sLocal, 8);
      float scale = 1.0f / sLocal.GetValue(0);

      // Step 6: y = x * scale.
      AscendC::Muls(yLocal, xLocal, scale, dimLenAlign);

      // Step 7: y = y * weight.
      AscendC::Mul(yLocal, yLocal, wLocal, dimLenAlign);

    } else {
      // ---- FP16 path: cast → fp32 compute → cast back. ----
      AscendC::LocalTensor<float> xF32 = xFp32Buf.Get<float>();
      AscendC::LocalTensor<float> tmpF32 = tmpFp32Buf.Get<float>();

      // Cast input fp16 → fp32.
      AscendC::Cast(xF32, xLocal, AscendC::RoundMode::CAST_NONE, dimLenAlign);

      // Step 1: x^2 in fp32.
      AscendC::Mul(tmpF32, xF32, xF32, dimLenAlign);

      // Step 2: ReduceSum(x^2) -> sLocal[0].
      AscendC::ReduceSum(sLocal, tmpF32, rTmp, dimLenAlign);

      // Step 3-5: scale = 1 / sqrt(mean(x^2) + eps).
      float sumVal = sLocal.GetValue(0);
      float meanVal = sumVal / static_cast<float>(dimLen) + this->eps;
      sLocal.SetValue(0, meanVal);
      AscendC::Sqrt(sLocal, sLocal, 8);
      float scale = 1.0f / sLocal.GetValue(0);

      // Step 6: y = x * scale (fp32).
      AscendC::Muls(tmpF32, xF32, scale, dimLenAlign);

      // Step 7: y = y * weight (fp32).
      AscendC::Mul(tmpF32, tmpF32, wLocal, dimLenAlign);

      // Cast result fp32 → fp16.
      AscendC::Cast(yLocal, tmpF32, AscendC::RoundMode::CAST_ROUND,
                    dimLenAlign);
    }

    inQueueX.FreeTensor(xLocal);
    outQueueY.EnQue(yLocal);
  }

  __aicore__ inline void CopyOut(int64_t row) {
    AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(this->dimLengthAlign * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(yGm[row * this->dimLengthAlign], yLocal, params);
    outQueueY.FreeTensor(yLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;

  AscendC::TBuf<AscendC::TPosition::VECCALC> weightBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32Buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpFp32Buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;

  AscendC::GlobalTensor<T> xGm, yGm;
  AscendC::GlobalTensor<float> weightGm;

  int64_t blockRows;
  int64_t dimLength;
  int64_t dimLengthAlign;
  float eps;
};

extern "C" __global__ __aicore__ void rms_norm(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, int64_t totalRows, int64_t dimLength,
    int64_t dimLengthAlign, int64_t formerNum, int64_t formerLength,
    int64_t tailLength, float eps, int64_t dtypeSize) {
  if (dtypeSize == 2) {
    KernelRmsNorm<half> op;
    op.Init(x, weight, y, totalRows, dimLength, dimLengthAlign, formerNum,
            formerLength, tailLength, eps);
    op.Process();
  } else {
    KernelRmsNorm<float> op;
    op.Init(x, weight, y, totalRows, dimLength, dimLengthAlign, formerNum,
            formerLength, tailLength, eps);
    op.Process();
  }
}
