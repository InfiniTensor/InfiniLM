#include "kernel_operator.h"

class KernelReshapeAndCacheFlashStrided {
 public:
  __aicore__ inline KernelReshapeAndCacheFlashStrided() {}

  __aicore__ inline void Init(
      GM_ADDR key, GM_ADDR value, GM_ADDR slot_mapping, GM_ADDR key_cache,
      GM_ADDR value_cache, int64_t num_tokens, int64_t num_heads,
      int64_t head_size, int64_t block_size, int64_t num_blocks,
      int64_t key_token_stride, int64_t key_head_stride,
      int64_t value_token_stride, int64_t value_head_stride,
      int64_t key_cache_block_stride, int64_t key_cache_page_stride,
      int64_t key_cache_head_stride, int64_t value_cache_block_stride,
      int64_t value_cache_page_stride, int64_t value_cache_head_stride,
      int64_t element_size) {
    key_.SetGlobalBuffer((__gm__ uint8_t*)key);
    value_.SetGlobalBuffer((__gm__ uint8_t*)value);
    slot_mapping_.SetGlobalBuffer((__gm__ int64_t*)slot_mapping);
    key_cache_.SetGlobalBuffer((__gm__ uint8_t*)key_cache);
    value_cache_.SetGlobalBuffer((__gm__ uint8_t*)value_cache);

    num_tokens_ = num_tokens;
    num_heads_ = num_heads;
    head_bytes_ = head_size * element_size;
    block_size_ = block_size;
    num_blocks_ = num_blocks;
    element_size_ = element_size;

    key_token_stride_ = key_token_stride;
    key_head_stride_ = key_head_stride;
    value_token_stride_ = value_token_stride;
    value_head_stride_ = value_head_stride;
    key_cache_block_stride_ = key_cache_block_stride;
    key_cache_page_stride_ = key_cache_page_stride;
    key_cache_head_stride_ = key_cache_head_stride;
    value_cache_block_stride_ = value_cache_block_stride;
    value_cache_page_stride_ = value_cache_page_stride;
    value_cache_head_stride_ = value_cache_head_stride;

    int64_t aligned_bytes = (head_bytes_ + 31) / 32 * 32;
    pipe_.InitBuffer(copy_buffer_, static_cast<uint32_t>(aligned_bytes));
  }

  __aicore__ inline void Process() {
    auto local = copy_buffer_.Get<uint8_t>();
    AscendC::DataCopyExtParams copy_params{
        1, static_cast<uint32_t>(head_bytes_), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> pad_params{false, 0, 0, 0};

    for (int64_t token = 0; token < num_tokens_; ++token) {
      int64_t slot = slot_mapping_.GetValue(token);
      if (slot < 0 || slot >= num_blocks_ * block_size_) {
        continue;
      }

      int64_t block = slot / block_size_;
      int64_t page = slot % block_size_;

      for (int64_t head = 0; head < num_heads_; ++head) {
        int64_t key_offset =
            (token * key_token_stride_ + head * key_head_stride_) *
            element_size_;
        int64_t key_cache_offset =
            (block * key_cache_block_stride_ + page * key_cache_page_stride_ +
             head * key_cache_head_stride_) *
            element_size_;
        Copy(key_, key_offset, key_cache_, key_cache_offset, local, copy_params,
             pad_params);

        int64_t value_offset =
            (token * value_token_stride_ + head * value_head_stride_) *
            element_size_;
        int64_t value_cache_offset = (block * value_cache_block_stride_ +
                                      page * value_cache_page_stride_ +
                                      head * value_cache_head_stride_) *
                                     element_size_;
        Copy(value_, value_offset, value_cache_, value_cache_offset, local,
             copy_params, pad_params);
      }
    }
  }

 private:
  __aicore__ inline void Copy(
      AscendC::GlobalTensor<uint8_t>& source, int64_t source_offset,
      AscendC::GlobalTensor<uint8_t>& destination, int64_t destination_offset,
      AscendC::LocalTensor<uint8_t>& local,
      const AscendC::DataCopyExtParams& copy_params,
      const AscendC::DataCopyPadExtParams<uint8_t>& pad_params) {
    AscendC::DataCopyPad(local, source[source_offset], copy_params, pad_params);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(destination[destination_offset], local, copy_params);
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> copy_buffer_;
  AscendC::GlobalTensor<uint8_t> key_;
  AscendC::GlobalTensor<uint8_t> value_;
  AscendC::GlobalTensor<int64_t> slot_mapping_;
  AscendC::GlobalTensor<uint8_t> key_cache_;
  AscendC::GlobalTensor<uint8_t> value_cache_;

  int64_t num_tokens_{0};
  int64_t num_heads_{0};
  int64_t head_bytes_{0};
  int64_t block_size_{0};
  int64_t num_blocks_{0};
  int64_t element_size_{0};
  int64_t key_token_stride_{0};
  int64_t key_head_stride_{0};
  int64_t value_token_stride_{0};
  int64_t value_head_stride_{0};
  int64_t key_cache_block_stride_{0};
  int64_t key_cache_page_stride_{0};
  int64_t key_cache_head_stride_{0};
  int64_t value_cache_block_stride_{0};
  int64_t value_cache_page_stride_{0};
  int64_t value_cache_head_stride_{0};
};

extern "C" __global__ __aicore__ void reshape_and_cache_flash_strided(
    GM_ADDR key, GM_ADDR value, GM_ADDR slot_mapping, GM_ADDR key_cache,
    GM_ADDR value_cache, int64_t num_tokens, int64_t num_heads,
    int64_t head_size, int64_t block_size, int64_t num_blocks,
    int64_t key_token_stride, int64_t key_head_stride,
    int64_t value_token_stride, int64_t value_head_stride,
    int64_t key_cache_block_stride, int64_t key_cache_page_stride,
    int64_t key_cache_head_stride, int64_t value_cache_block_stride,
    int64_t value_cache_page_stride, int64_t value_cache_head_stride,
    int64_t element_size) {
  KernelReshapeAndCacheFlashStrided op;
  op.Init(key, value, slot_mapping, key_cache, value_cache, num_tokens,
          num_heads, head_size, block_size, num_blocks, key_token_stride,
          key_head_stride, value_token_stride, value_head_stride,
          key_cache_block_stride, key_cache_page_stride, key_cache_head_stride,
          value_cache_block_stride, value_cache_page_stride,
          value_cache_head_stride, element_size);
  op.Process();
}
