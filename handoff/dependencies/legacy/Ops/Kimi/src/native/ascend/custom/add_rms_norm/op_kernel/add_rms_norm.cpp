#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelAddRmsNorm {
 public:
  __aicore__ inline KernelAddRmsNorm() {}

  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y,
                              GM_ADDR x_out, int64_t totalRows,
                              int64_t dimLength, int64_t dimLengthAlign,
                              int64_t formerNum, int64_t formerLength,
                              int64_t tailLength, float eps) {
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
    x1Gm.SetGlobalBuffer((__gm__ T*)x1 + rowOffset * dimLengthAlign,
                         this->blockRows * dimLengthAlign);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2 + rowOffset * dimLengthAlign,
                         this->blockRows * dimLengthAlign);
    yGm.SetGlobalBuffer((__gm__ T*)y + rowOffset * dimLengthAlign,
                        this->blockRows * dimLengthAlign);
    xOutGm.SetGlobalBuffer((__gm__ T*)x_out + rowOffset * dimLengthAlign,
                           this->blockRows * dimLengthAlign);
    weightGm.SetGlobalBuffer((__gm__ float*)weight, dimLengthAlign);

    int32_t dimLenAlign = static_cast<int32_t>(this->dimLengthAlign);

    // I/O queues (double-buffered).
    pipe.InitBuffer(inQueueX1, BUFFER_NUM,
                    dimLenAlign * static_cast<int32_t>(sizeof(T)));
    pipe.InitBuffer(inQueueX2, BUFFER_NUM,
                    dimLenAlign * static_cast<int32_t>(sizeof(T)));
    pipe.InitBuffer(outQueueY, BUFFER_NUM,
                    dimLenAlign * static_cast<int32_t>(sizeof(T)));
    pipe.InitBuffer(outQueueXOut, BUFFER_NUM,
                    dimLenAlign * static_cast<int32_t>(sizeof(T)));

    // Weight buffer (fp32, loaded once, reused for all rows).
    pipe.InitBuffer(weightBuf,
                    dimLenAlign * static_cast<int32_t>(sizeof(float)));

    // FP16 path needs extra fp32 compute buffers.
    // buf1: holds x_out in fp32 (reused from x1_fp32 after Add).
    // buf2: holds x2_fp32 initially, then x_out^2, then final result.
    if constexpr (sizeof(T) == 2) {
      pipe.InitBuffer(fp32Buf1,
                      dimLenAlign * static_cast<int32_t>(sizeof(float)));
      pipe.InitBuffer(fp32Buf2,
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
    AscendC::LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
    AscendC::LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(this->dimLengthAlign * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(x1Local, x1Gm[row * this->dimLengthAlign], params,
                         pad);
    AscendC::DataCopyPad(x2Local, x2Gm[row * this->dimLengthAlign], params,
                         pad);
    inQueueX1.EnQue(x1Local);
    inQueueX2.EnQue(x2Local);
  }

  __aicore__ inline void Compute(int64_t row) {
    AscendC::LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
    AscendC::LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<T> xOutLocal = outQueueXOut.AllocTensor<T>();

    AscendC::LocalTensor<float> wLocal = weightBuf.Get<float>();
    AscendC::LocalTensor<float> rTmp = reduceTmpBuf.Get<float>();
    AscendC::LocalTensor<float> sLocal = sumBuf.Get<float>();

    int32_t dimLen = static_cast<int32_t>(this->dimLength);
    int32_t dimLenAlign = static_cast<int32_t>(this->dimLengthAlign);

    if constexpr (sizeof(T) == 4) {
      // ---- FP32 path: compute directly. ----

      // Step 1: x_out = x1 + x2.
      AscendC::Add(xOutLocal, x1Local, x2Local, dimLenAlign);

      // Step 2: x_out^2 into yLocal (reuse output buffer temporarily).
      AscendC::Mul(yLocal, xOutLocal, xOutLocal, dimLenAlign);

      // Step 3: ReduceSum(x_out^2) -> sLocal[0].
      // ReduceSum may modify yLocal, but we overwrite it below.
      AscendC::ReduceSum(sLocal, yLocal, rTmp, dimLenAlign);

      // Step 4-5: scale = 1 / sqrt(mean(x_out^2) + eps).
      float sumVal = sLocal.GetValue(0);
      float meanVal = sumVal / static_cast<float>(dimLen) + this->eps;
      sLocal.SetValue(0, meanVal);
      AscendC::Sqrt(sLocal, sLocal, 8);
      float scale = 1.0f / sLocal.GetValue(0);

      // Step 6: y = x_out * scale.
      AscendC::Muls(yLocal, xOutLocal, scale, dimLenAlign);

      // Step 7: y = y * weight.
      AscendC::Mul(yLocal, yLocal, wLocal, dimLenAlign);

    } else {
      // ---- FP16 path: cast → fp32 compute → cast back. ----
      AscendC::LocalTensor<float> b1 = fp32Buf1.Get<float>();
      AscendC::LocalTensor<float> b2 = fp32Buf2.Get<float>();

      // Cast inputs fp16 → fp32.
      AscendC::Cast(b1, x1Local, AscendC::RoundMode::CAST_NONE, dimLenAlign);
      AscendC::Cast(b2, x2Local, AscendC::RoundMode::CAST_NONE, dimLenAlign);

      // Step 1: x_out = x1 + x2 (fp32), stored in b1.
      AscendC::Add(b1, b1, b2, dimLenAlign);

      // Cast x_out fp32 → fp16 for the x_out output.
      AscendC::Cast(xOutLocal, b1, AscendC::RoundMode::CAST_ROUND, dimLenAlign);

      // Step 2: x_out^2 in fp32, stored in b2.
      AscendC::Mul(b2, b1, b1, dimLenAlign);

      // Step 3: ReduceSum(x_out^2) -> sLocal[0].
      AscendC::ReduceSum(sLocal, b2, rTmp, dimLenAlign);

      // Step 4-5: scale = 1 / sqrt(mean(x_out^2) + eps).
      float sumVal = sLocal.GetValue(0);
      float meanVal = sumVal / static_cast<float>(dimLen) + this->eps;
      sLocal.SetValue(0, meanVal);
      AscendC::Sqrt(sLocal, sLocal, 8);
      float scale = 1.0f / sLocal.GetValue(0);

      // Step 6: y = x_out * scale (fp32), reuse b2.
      AscendC::Muls(b2, b1, scale, dimLenAlign);

      // Step 7: y = y * weight (fp32).
      AscendC::Mul(b2, b2, wLocal, dimLenAlign);

      // Cast result fp32 → fp16.
      AscendC::Cast(yLocal, b2, AscendC::RoundMode::CAST_ROUND, dimLenAlign);
    }

    inQueueX1.FreeTensor(x1Local);
    inQueueX2.FreeTensor(x2Local);
    outQueueY.EnQue(yLocal);
    outQueueXOut.EnQue(xOutLocal);
  }

  __aicore__ inline void CopyOut(int64_t row) {
    AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    AscendC::LocalTensor<T> xOutLocal = outQueueXOut.DeQue<T>();
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(this->dimLengthAlign * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(yGm[row * this->dimLengthAlign], yLocal, params);
    AscendC::DataCopyPad(xOutGm[row * this->dimLengthAlign], xOutLocal, params);
    outQueueY.FreeTensor(yLocal);
    outQueueXOut.FreeTensor(xOutLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX1;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX2;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueXOut;

  AscendC::TBuf<AscendC::TPosition::VECCALC> weightBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp32Buf1;
  AscendC::TBuf<AscendC::TPosition::VECCALC> fp32Buf2;
  AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;

  AscendC::GlobalTensor<T> x1Gm, x2Gm, yGm, xOutGm;
  AscendC::GlobalTensor<float> weightGm;

  int64_t blockRows;
  int64_t dimLength;
  int64_t dimLengthAlign;
  float eps;
};

extern "C" __global__ __aicore__ void add_rms_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y, GM_ADDR x_out,
    int64_t totalRows, int64_t dimLength, int64_t dimLengthAlign,
    int64_t formerNum, int64_t formerLength, int64_t tailLength, float eps,
    int64_t dtypeSize) {
  if (dtypeSize == 2) {
    KernelAddRmsNorm<half> op;
    op.Init(x1, x2, weight, y, x_out, totalRows, dimLength, dimLengthAlign,
            formerNum, formerLength, tailLength, eps);
    op.Process();
  } else {
    KernelAddRmsNorm<float> op;
    op.Init(x1, x2, weight, y, x_out, totalRows, dimLength, dimLengthAlign,
            formerNum, formerLength, tailLength, eps);
    op.Process();
  }
}
