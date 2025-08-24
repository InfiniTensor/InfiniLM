

#include "qwen3_impl.hpp"
#include "qwen3_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <random>
#include <sys/stat.h>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

/*
 * Debug utilities for comparing C++ vs Python implementations
 * These functions save tensor data to files for comparison
 */

// Global debug flag - set to true to enable debug output
static bool g_debug_enabled = true;

void set_debug_mode(bool enabled) { g_debug_enabled = enabled; }

// Validation function to check tensor for extreme values
bool validate_tensor_range(const std::shared_ptr<Tensor> &tensor,
                           const std::string &name, float min_threshold = -1e6,
                           float max_threshold = 1e6) {
  if (!tensor)
    return false;

  auto shape = tensor->shape();
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }

  // For small tensors, check all values; for large tensors, sample
  size_t sample_size = std::min(total_size, size_t(1000));
  std::vector<float> host_data(sample_size);

  // Copy sample data to host
  RUN_INFINI(infinirtMemcpy(host_data.data(), tensor->data(),
                            sample_size * sizeof(float), INFINIRT_MEMCPY_D2H));
  RUN_INFINI(infinirtDeviceSynchronize());

  // Check for extreme values
  bool has_extreme = false;
  float actual_min = host_data[0], actual_max = host_data[0];
  int inf_count = 0, nan_count = 0;

  for (size_t i = 0; i < sample_size; ++i) {
    float val = host_data[i];
    if (std::isnan(val)) {
      nan_count++;
      has_extreme = true;
    } else if (std::isinf(val)) {
      inf_count++;
      has_extreme = true;
    } else {
      actual_min = std::min(actual_min, val);
      actual_max = std::max(actual_max, val);
      if (val < min_threshold || val > max_threshold) {
        has_extreme = true;
      }
    }
  }

  if (has_extreme && g_debug_enabled) {
    printf("⚠ RANGE WARNING: %s has extreme values:\n", name.c_str());
    printf("  Min: %e, Max: %e\n", actual_min, actual_max);
    printf("  NaN count: %d, Inf count: %d (in sample of %zu)\n", nan_count,
           inf_count, sample_size);
  }

  return !has_extreme;
}

// Function to clamp tensor values to prevent overflow in subsequent operations
void clamp_tensor_inplace(const std::shared_ptr<Tensor> &tensor,
                          float min_val = -65504.0f, float max_val = 65504.0f) {
  if (!tensor)
    return;

  auto shape = tensor->shape();
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }

  // Get data as float pointer (assuming FP32 storage for intermediate
  // computations)
  float *data_ptr = static_cast<float *>(tensor->data());

  // Launch a simple clamping operation on device (this would need proper
  // CUDA/device implementation) For now, copy to host, clamp, and copy back
  // (inefficient but safe)
  std::vector<float> host_data(total_size);

  // Copy to host
  RUN_INFINI(infinirtMemcpy(host_data.data(), data_ptr,
                            total_size * sizeof(float), INFINIRT_MEMCPY_D2H));
  RUN_INFINI(infinirtDeviceSynchronize());

  // Clamp values
  bool clamped_any = false;
  for (size_t i = 0; i < total_size; ++i) {
    if (host_data[i] < min_val) {
      host_data[i] = min_val;
      clamped_any = true;
    } else if (host_data[i] > max_val) {
      host_data[i] = max_val;
      clamped_any = true;
    } else if (std::isnan(host_data[i]) || std::isinf(host_data[i])) {
      host_data[i] = 0.0f; // Replace NaN/Inf with zero
      clamped_any = true;
    }
  }

  if (clamped_any) {
    // Copy back to device
    RUN_INFINI(infinirtMemcpy(data_ptr, host_data.data(),
                              total_size * sizeof(float), INFINIRT_MEMCPY_H2D));
    RUN_INFINI(infinirtDeviceSynchronize());

    if (g_debug_enabled) {
      printf("⚠ Clamped extreme values in tensor to range [%f, %f]\n", min_val,
             max_val);
    }
  }
}

template <typename T>
void save_tensor_debug(const std::shared_ptr<Tensor> &tensor,
                       const std::string &name, int layer = -1,
                       const std::string &prefix = "cpp") {
  if (!g_debug_enabled || !tensor)
    return;

  // Create filename
  std::string filename;
  if (layer >= 0) {
    filename = "output/" + prefix + "_layer_" + std::to_string(layer) + "_" +
               name + ".txt";
  } else {
    filename = "output/" + prefix + "_" + name + ".txt";
  }

  // Get tensor data to CPU
  auto shape = tensor->shape();
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }

  // Allocate host memory for tensor data
  std::vector<T> host_data(total_size);

  // Copy from device to host
  RUN_INFINI(infinirtMemcpy(host_data.data(), tensor->data(),
                            total_size * sizeof(T), INFINIRT_MEMCPY_D2H));

  // Synchronize to ensure copy is complete
  RUN_INFINI(infinirtDeviceSynchronize());

  // Calculate comprehensive statistics
  T mean = 0, min_val = host_data[0], max_val = host_data[0];
  int inf_count = 0, nan_count = 0, zero_count = 0;
  double abs_sum = 0.0;

  for (size_t i = 0; i < total_size; ++i) {
    T val = host_data[i];
    if (std::isnan(val)) {
      nan_count++;
    } else if (std::isinf(val)) {
      inf_count++;
    } else if (val == 0) {
      zero_count++;
      mean += val; // still count in mean
    } else {
      mean += val;
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
      abs_sum += std::abs(static_cast<double>(val));
    }
  }
  mean /= total_size;

  T std_dev = 0;
  for (size_t i = 0; i < total_size; ++i) {
    if (!std::isnan(host_data[i]) && !std::isinf(host_data[i])) {
      T diff = host_data[i] - mean;
      std_dev += diff * diff;
    }
  }
  std_dev = std::sqrt(std_dev / total_size);

  // Save to file
  std::ofstream file(filename);
  file << std::scientific << std::setprecision(6);
  file << "# Tensor: " << name << "\n";
  file << "# Shape: ";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      file << "x";
    file << shape[i];
  }
  file << "\n";
  file << "# Total elements: " << total_size << "\n";
  file << "# Data type size: " << sizeof(T) << " bytes\n";
  file << "# Statistics:\n";
  file << "# Mean: " << mean << "\n";
  file << "# Std: " << std_dev << "\n";
  file << "# Min: " << min_val << "\n";
  file << "# Max: " << max_val << "\n";
  file << "# L1 norm (avg abs): " << (abs_sum / total_size) << "\n";
  file << "# Special values - NaN: " << nan_count << ", Inf: " << inf_count
       << ", Zero: " << zero_count << "\n";

  // Detect potential issues
  bool has_issues = false;
  if (nan_count > 0 || inf_count > 0) {
    file << "# ⚠ WARNING: Contains NaN or Inf values!\n";
    has_issues = true;
  }
  if (std::abs(mean) > 1e6) {
    file << "# ⚠ WARNING: Mean is extremely large (>" << 1e6 << ")\n";
    has_issues = true;
  }
  if (std::abs(max_val) > 1e6 || std::abs(min_val) > 1e6) {
    file << "# ⚠ WARNING: Contains values beyond FP16 safe range\n";
    has_issues = true;
  }
  if (zero_count > total_size * 0.9) {
    file
        << "# ⚠ WARNING: More than 90% values are zero (potential underflow)\n";
    has_issues = true;
  }

  if (!has_issues) {
    file << "# ✓ Values appear to be in reasonable range\n";
  }

  file << "# Data:\n";

  // For large tensors, save first and last elements only
  if (total_size > 1000) {
    file << "# First 50 elements:\n";
    for (size_t i = 0; i < std::min(size_t(50), total_size); ++i) {
      file << host_data[i] << "\n";
    }
    file << "# Last 50 elements:\n";
    for (size_t i = std::max(size_t(0), total_size - 50); i < total_size; ++i) {
      file << host_data[i] << "\n";
    }
  } else {
    // Save all elements for small tensors
    for (size_t i = 0; i < total_size; ++i) {
      file << host_data[i] << "\n";
    }
  }

  file.close();

  // Print summary to console
  if (has_issues) {
    printf("⚠ Tensor %s has data quality issues - check %s\n", name.c_str(),
           filename.c_str());
  }
}

// Helper function to convert FP16 to float
float fp16_to_float(uint16_t fp16_val) {
  uint32_t sign = (fp16_val & 0x8000) << 16;     // 符号位
  uint32_t exponent = (fp16_val & 0x7C00) >> 10; // 指数位 (5位)
  uint32_t mantissa = (fp16_val & 0x03FF);       // 尾数位 (10位)

  uint32_t float_bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      // Zero
      float_bits = sign;
    } else {
      // Denormalized number
      exponent = 127 - 14; // FP32 bias - FP16 bias - 1
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3FF;
      float_bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    // Infinity or NaN
    float_bits = sign | 0x7F800000 | (mantissa << 13);
  } else {
    // Normalized number
    exponent = exponent - 15 + 127; // 转换偏置：FP16偏置15 -> FP32偏置127
    float_bits = sign | (exponent << 23) | (mantissa << 13);
  }

  return *reinterpret_cast<float *>(&float_bits);
}

void save_tensor_debug_f16(const std::shared_ptr<Tensor> &tensor,
                           const std::string &name, int layer = -1,
                           const std::string &prefix = "cpp") {
  if (!g_debug_enabled || !tensor)
    return;

  // Create filename
  std::string filename;
  if (layer >= 0) {
    filename = "output/" + prefix + "_layer_" + std::to_string(layer) + "_" +
               name + ".txt";
  } else {
    filename = "output/" + prefix + "_" + name + ".txt";
  }

  // Get tensor data to CPU
  auto shape = tensor->shape();
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }

  // 按 FP16 读取数据
  std::vector<uint16_t> host_data_raw(total_size);

  // Copy from device to host as FP16
  RUN_INFINI(infinirtMemcpy(host_data_raw.data(), tensor->data(),
                            total_size * sizeof(uint16_t),
                            INFINIRT_MEMCPY_D2H));

  // Synchronize to ensure copy is complete
  RUN_INFINI(infinirtDeviceSynchronize());

  // Convert FP16 to float for analysis
  std::vector<float> host_data(total_size);
  for (size_t i = 0; i < total_size; i++) {
    host_data[i] = fp16_to_float(host_data_raw[i]); // 使用正确的转换函数
  }

  // Calculate statistics on converted data
  float mean = 0, min_val = host_data[0], max_val = host_data[0];
  for (size_t i = 0; i < total_size; ++i) {
    mean += host_data[i];
    min_val = std::min(min_val, host_data[i]);
    max_val = std::max(max_val, host_data[i]);
  }
  mean /= total_size;

  float std_dev = 0;
  for (size_t i = 0; i < total_size; ++i) {
    float diff = host_data[i] - mean;
    std_dev += diff * diff;
  }
  std_dev = std::sqrt(std_dev / total_size);

  // Save to file with both raw hex and converted values
  std::ofstream file(filename);
  file << std::scientific << std::setprecision(6);
  file << "# Tensor: " << name << " (FP16 format)\n";
  file << "# Shape: ";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      file << "x";
    file << shape[i];
  }
  file << "\n";
  file << "# Total elements: " << total_size << "\n";
  file << "# Mean: " << mean << " (converted)\n";
  file << "# Std: " << std_dev << " (converted)\n";
  file << "# Min: " << min_val << " (converted)\n";
  file << "# Max: " << max_val << " (converted)\n";
  file << "# Data (first 100 elements as hex and converted):\n";

  // Save first 100 elements showing both raw hex and converted values
  for (size_t i = 0; i < std::min(size_t(100), total_size); ++i) {
    file << "0x" << std::hex << std::setw(4) << std::setfill('0')
         << host_data_raw[i] << " (" << std::scientific << host_data[i]
         << ")\n";
  }

  file.close();
  printf("  Saved FP16 debug tensor: %s\n", filename.c_str());
}
void save_tensor_debug_f32(const std::shared_ptr<Tensor> &tensor,
                           const std::string &name, int layer = -1,
                           const std::string &prefix = "cpp") {
  save_tensor_debug<float>(tensor, name, layer, prefix);
}

/*
 * 设备资源创建和初始化
 *
 * 创建和初始化推理所需的所有 GPU/设备资源，包括：
 * - InfiniCore 设备上下文和操作句柄
 * - 用于多设备并行的分布式张量权重
 * - 用于高效缓冲区管理的内存池
 * - 用于设备间同步的通信上下文
 *
 * 参数：
 * - rsrc：要填充的输出设备资源结构
 * - meta：模型元数据（层数、维度、数据类型）
 * - weights：模型权重张量
 * - device：InfiniCore 设设备类型（GPU/CPU）
 * - idev：分布式设置中的当前设备索引（0 到 ndev-1）
 * - ndev：用于张量并行的设备总数
 * - dev_id：物理设备 ID
 * - comm：用于多设备操作的 InfiniCCL 通信器
 */
void createQwen3DeviceResource(DeviceQwen3Resource *rsrc, const Qwen3Meta *meta,
                               const Qwen3Weights *weights,
                               infiniDevice_t device, int idev, int ndev,
                               int dev_id, infinicclComm_t comm) {
  // 初始化 InfiniCore 设备上下文并创建操作句柄
  // 这设置了后续 InfiniCore API 调用的活动设备
  RUN_INFINI(infinirtSetDevice(device, dev_id));

  // 为此设备创建操作句柄 - 用于所有计算操作
  infiniopHandle_t handle;
  infiniopCreateHandle(&handle);

  // 创建用于异步操作的执行流
  infinirtStream_t stream;
  infinirtStreamCreate(&stream);

  /*
   * 用于分布式推理的权重张量提取
   *
   * 从全局权重存储中提取模型权重并在设备间分区
   * 以实现张量并行。每个设备获得：
   * - 注意力投影权重（QKV，输出）：按注意力头分区
   * - FFN 权重：按中间维度分区
   * - 归一化权重：在所有设备上复制
   *
   * 张量形状：
   * - w_attn_norm：[d] - 层归一化权重，复制
   * - w_attn_qkv：[d, (nh + 2*nkvh)/ndev * dh] - QKV 投影，头分区
   * - b_attn_qkv：[(nh + 2*nkvh)/ndev * dh] - QKV 偏置（可选），头分区
   * - w_attn_out：[nh/ndev * dh, d] - 输出投影，头分区
   * - w_ffn_norm：[d] - FFN 归一化权重，复制
   * - w_ffn_gate_up：[d, 2*di/ndev] - 门控和上升投影，维度分区
   * - w_ffn_down：[di/ndev, d] - 下降投影，维度分区
   */

  std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_q_norm,
      w_attn_k_norm, w_attn_v_norm, w_attn_q_proj, w_attn_k_proj, w_attn_v_proj,
      w_attn_o_proj, w_mlp_norm, w_mlp_gate_proj, w_mlp_up_proj,
      w_mlp_down_proj;

  // 新的分离权重提取方式
  for (size_t layer = 0; layer < meta->nlayer; layer++) {

    // 添加注意力层归一化权重提取（这个被遗漏了！）
    w_attn_norm.push_back(
        getQwen3AttnNorm(meta, weights, layer)); // [d] - 注意力层归一化权重

    // Qwen3 特有：Q/K 归一化权重（每层每头维度）

    w_attn_q_norm.push_back(
        getQwen3QNorm(meta, weights, layer)); // [dh] - Q 归一化权重
    w_attn_k_norm.push_back(
        getQwen3KNorm(meta, weights, layer)); // [dh] - K 归一化权重

    // 分离的 QKV 投影权重
    // Q 投影权重：[d, nh/ndev * dh] - 按查询头分区
    w_attn_q_proj.push_back(getQwen3AttnQ(meta, weights, layer, idev, ndev));

    // K 投影权重：[d, nkvh/ndev * dh] - 按键值头分区
    w_attn_k_proj.push_back(getQwen3AttnK(meta, weights, layer, idev, ndev));

    // V 投影权重：[d, nkvh/ndev * dh] - 按键值头分区
    w_attn_v_proj.push_back(getQwen3AttnV(meta, weights, layer, idev, ndev));

    // 注意力输出投影：[nh/ndev * dh, d] - 按输入维度分区
    w_attn_o_proj.push_back(getQwen3AttnO(meta, weights, layer, idev, ndev));

    // MLP 层权重

    // MLP 前归一化：[d] - 在所有设备上复制
    w_mlp_norm.push_back(getQwen3MLPNorm(meta, weights, layer));

    // MLP Gate 投影：[d, di/ndev] - 按中间维度分区
    w_mlp_gate_proj.push_back(
        getQwen3MLPGate(meta, weights, layer, idev, ndev));

    // MLP Up 投影：[d, di/ndev] - 按中间维度分区
    w_mlp_up_proj.push_back(getQwen3MLPUp(meta, weights, layer, idev, ndev));

    // MLP Down 投影：[di/ndev, d] - 按输入维度分区
    w_mlp_down_proj.push_back(
        getQwen3MLPDown(meta, weights, layer, idev, ndev));
  }

  // 创建用于高效缓冲区分配的内存池（128MB）
  // 此池在推理期间管理临时张量以避免频繁的 malloc/free
  // 增加内存池大小
  auto memory_pool =
      std::make_shared<MemoryPool>(512 * 1024 * 1024); // 从128MB增加到512MB
  // 使用所有初始化的组件填充设备资源结构
  // 此结构包含在此设备上推理所需的一切
  *rsrc = DeviceQwen3Resource{
      device,                         // InfiniCore 设备类型（GPU/CPU）
      dev_id,                         // 物理设备 ID
      handle,                         // InfiniCore 操作句柄
      getQwen3InEmbd(meta, weights),  // 输入嵌入表 [dvoc, d]
      getQwen3OutNorm(meta, weights), // 输出归一化权重 [d]
      getQwen3OutEmbd(meta, weights), // 输出嵌入/LM 头 [d, dvoc]
      getQwen3SinTable(meta),         // RoPE 正弦表 [dctx, dh/2]
      getQwen3CosTable(meta),         // RoPE 余弦表 [dctx, dh/2]
      w_attn_norm,                    // 注意力层归一化权重 [d]
      w_attn_q_norm,                  // Q 归一化权重
      w_attn_k_norm,                  // K 归一化权重
      w_attn_q_proj,                  // Q 投影权重
      w_attn_k_proj,                  // K 投影权重
      w_attn_v_proj,                  // V 投影权重
      w_attn_o_proj,                  // 注意力输出权重
      w_mlp_norm,                     // MLP 归一化权重
      w_mlp_gate_proj,                // MLP Gate 权重
      w_mlp_up_proj,                  // MLP Up 权重
      w_mlp_down_proj,                // MLP Down 权重
      stream,                         // 用于异步操作的执行流
      comm,                           // 设备间通信上下文
      memory_pool,                    // 用于临时缓冲区的内存池
  };

  // 同步设备以确保所有初始化完成
  RUN_INFINI(infinirtDeviceSynchronize());
}

/*
 * 设备资源清理和内存释放
 *
 * 按分配的相反顺序正确释放所有设备资源：
 * 1. 同步设备以完成所有待处理操作
 * 2. 释放张量内存（shared_ptr 自动处理引用计数）
 * 3. 销毁 InfiniCore 句柄和流
 * 4. 清理通信上下文
 *
 * 这可以防止内存泄漏并确保正确清理 GPU 资源
 */
void releaseQwen3DeviceResource(DeviceQwen3Resource &res) {
  // 在清理前等待所有待处理操作完成
  infinirtDeviceSynchronize();

  // 通过重置 shared_ptr 引用来释放张量内存
  // 当引用计数达到零时，底层内存将被释放
  // 释放全局模型张量（输入/输出嵌入、归一化、RoPE 表）
  res.w_in_embd.reset();  // 输入嵌入表 [dvoc, d]
  res.w_out_norm.reset(); // 最终层归一化 [d]
  res.w_out_embd.reset(); // 输出投影/LM 头 [d, dvoc]
  res.sin_table.reset();  // RoPE 正弦查找表 [dctx, dh/2]
  res.cos_table.reset();  // RoPE 余弦查找表 [dctx, dh/2]

  // 释放每层注意力权重并清除向量
  for (auto &t : res.w_attn_q_norm) {
    t.reset(); // Q 层归一化权重 [d]
  }
  res.w_attn_q_norm.clear();
  for (auto &t : res.w_attn_k_norm) {
    t.reset(); // K 层归一化权重 [d]
  }
  res.w_attn_k_norm.clear();
  for (auto &t : res.w_attn_q_proj) {
    t.reset(); // Q 投影权重 [d, nh/ndev * dh]
  }
  res.w_attn_q_proj.clear();
  for (auto &t : res.w_attn_k_proj) {
    t.reset(); // K 投影权重 [d, nkvh/ndev * dh]
  }
  res.w_attn_k_proj.clear();
  for (auto &t : res.w_attn_v_proj) {
    t.reset(); // V 投影权重 [d, nkvh/ndev * dh]
  }
  res.w_attn_v_proj.clear();
  for (auto &t : res.w_attn_o_proj) {
    t.reset(); // 输出投影权重 [nh/ndev * dh, d]
  }
  res.w_attn_o_proj.clear();
  for (auto &t : res.w_mlp_norm) {
    t.reset(); // MLP 层归一化权重 [d]
  }
  res.w_mlp_norm.clear();
  for (auto &t : res.w_mlp_gate_proj) {
    t.reset(); // MLP Gate 投影权重 [d, di/ndev]
  }
  res.w_mlp_gate_proj.clear();
  for (auto &t : res.w_mlp_up_proj) {
    t.reset(); // MLP Up 投影权重 [d, di/ndev]
  }
  res.w_mlp_up_proj.clear();
  for (auto &t : res.w_mlp_down_proj) {
    t.reset(); // MLP Down 投影权重 [di/ndev, d]
  }
  res.w_mlp_down_proj.clear();

  // 销毁 InfiniCore 句柄和上下文
  infiniopDestroyHandle(res.handle); // 释放操作句柄
  res.handle = nullptr;

  infinirtStreamDestroy(res.stream); // 释放执行流
  res.stream = nullptr;

  infinicclCommDestroy(res.comm); // 释放通信上下文
  res.comm = nullptr;
}

/*
 * 设备级批处理推理函数
 * 在单个设备上为一批序列执行 transformer 推理。
 * 实现完整的前向传递，包括：
 *  1. 输入嵌入查找和 RoPE 位置编码
 *  2. 多层 transformer 块（注意力 + FFN）
 *  3. 输出归一化和概率分布
 *  4. 带温度/top-k/top-p 的 token 采样
 *
 * 此函数通过张量并行处理分布式推理，其中
 * 每个设备处理模型参数的一个切片。
 *
 * 输入参数：
 * - meta：模型架构元数据（维度、层数等）
 * - rsrc：设备资源（权重、句柄、内存池）
 * - idev/ndev：用于分布式推理的设备索引和设备总数
 * - tokens：要处理的输入 token ID [ntok]
 * - ntok：批处理中所有请求的 token 总数
 * - req_lens：每个请求的长度 [nreq]
 * - nreq：批处理中的请求数
 * - req_pos：每个请求在 KV 缓存中的起始位置 [nreq]
 * - kv_caches：每个请求的 KV 缓存存储 [nreq][ndev][nlayer]
 * - temperature/topk/topp：采样参数 [nreq]
 * - output：生成的 token ID [nreq]
 *
 * 张量维度符号：
 * - ntok：批处理中的 token 总数
 * - nreq：请求数
 * - d：模型隐藏维度
 * - nh：总注意力头数
 * - nkvh：总键值头数
 * - dh：头维度（d/nh）
 * - di：FFN 中间维度
 * - dvoc：词汇表大小
 * - dctx：最大上下文长度
 */
void inferQwen3DeviceBatch(const Qwen3Meta &meta, DeviceQwen3Resource &rsrc,
                           uint32_t idev, uint32_t ndev, const uint32_t *tokens,
                           uint32_t ntok, const uint32_t *req_lens,
                           uint32_t nreq, const uint32_t *req_pos,
                           struct Qwen3KVCache **kv_caches,
                           const float *temperature, const uint32_t *topk,
                           const float *topp, uint32_t *output) {

  if (meta.nh < ndev || meta.nkvh < ndev) {
    throw std::runtime_error(
        "Invalid distributed setup: heads (" + std::to_string(meta.nh) + ", " +
        std::to_string(meta.nkvh) + ") must be >= devices (" +
        std::to_string(ndev) + ")");
  }
  /*
   * 提取模型维度并配置分布式推理
   *
   * 张量并行的关键维度计算：
   * - nkvh：每设备的键值头数 = total_kv_heads / ndev
   * - nh：每设备的查询头数 = total_heads / ndev
   * - ngroup：分组查询注意力比率 = nh / nkvh
   * - di：每设备的 FFN 中间维度 = total_intermediate / ndev
   *
   * 这确保每个设备处理注意力头和 FFN 维度的一个切片
   * 同时保持相同的序列处理。
   */
  auto nlayer = meta.nlayer;    // transformer 层数
  auto nkvh = meta.nkvh / ndev; // 每设备的 KV 头数（分布式）
  auto nh = meta.nh / ndev;     // 每设备的查询头数（分布式）
  auto ngroup = nh / nkvh;      // 分组查询注意力因子
  // auto dctx = meta.dctx;           // 最大上下文长度（未使用）
  auto dh = meta.dh;               // 头维度
  auto d = meta.d;                 // 模型隐藏维度
  auto dt_logits = meta.dt_logits; // logits 的数据类型（FP16/BF16/FP32）
  auto di = meta.di / ndev;        // 每设备的 FFN 中间维度
  auto dvoc = meta.dvoc;           // 词汇表大小
  auto stream = rsrc.stream;       // 用于异步操作的执行流
  auto dt_logits_corrected = rsrc.w_in_embd->dtype(); // 使用权重的实际数据类型

  if (dt_logits != dt_logits_corrected) {
    if (g_debug_enabled && idev == 0) {
      printf("WARNING: Correcting dt_logits from %d to %d to match weights\n",
             dt_logits, dt_logits_corrected);
    }
    dt_logits = dt_logits_corrected;
  }

  if (nh == 0 || nkvh == 0) {
    throw std::runtime_error(
        "Zero heads after distribution - check model/device configuration");
  }

  if (ntok == 0 || nreq == 0) {
    throw std::runtime_error(
        "Invalid batch size: ntok=" + std::to_string(ntok) +
        ", nreq=" + std::to_string(nreq));
  }

  // 计算预期内存使用量
  size_t total_memory_needed = 0;
  total_memory_needed +=
      ntok * d * dsize(dt_logits) * 2; // logits_in + logits_out
  total_memory_needed +=
      ntok * nh * dh * dsize(dt_logits) * 3; // q_buf + q_norm_buf + o_buf
  total_memory_needed +=
      ntok * nkvh * dh * dsize(dt_logits) * 3; // k_buf + k_norm_buf + v_buf
  total_memory_needed += ntok * di * dsize(dt_logits) * 2; // gate_buf + up_buf
  total_memory_needed += nreq * dvoc * dsize(dt_logits);   // prob_buf
  total_memory_needed += nreq * sizeof(int64_t);           // result_buf

  if (!rsrc.memory_pool) {
    throw std::runtime_error("Memory pool is null");
  }

  /*
   * 推理流水线的内存缓冲区分配
   *
   * 为中间计算分配临时缓冲区。
   * 所有缓冲区使用设备内存池进行高效分配/释放。
   *
   * 缓冲区张量形状：
   * - logits_in/out：[ntok, d] - 流经各层的隐藏状态
   * - qkv_buf：[ntok, (nh + nkvh*2) * dh] - 连接的 Q、K、V 投影
   * - gate_up_buf：[ntok, 2*di] - 连接的 FFN 门控和上升投影
   * - o_buf：[ntok, nh*dh] - 注意力输出在输出投影之前
   * - prob_buf：[nreq, dvoc] - 输出概率分布
   * - result_buf：[nreq] - 采样的 token ID（设备内存）
   * - result_cpu：[nreq] - 采样的 token ID（主机内存用于输出）
   */

  auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
  assert(logits_in != nullptr && "logits_in allocation failed");

  if (!logits_in) {
    throw std::runtime_error("Failed to allocate logits_in buffer");
  }

  auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
  assert(logits_out != nullptr && "logits_out allocation failed");
  if (!logits_out) {
    throw std::runtime_error("Failed to allocate logits_out buffer");
  }

  auto q_buf = Tensor::buffer(dt_logits, {ntok, nh * dh},
                              rsrc.memory_pool); // Q 投影输出
  auto k_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh},
                              rsrc.memory_pool); // K 投影输出  、
  auto v_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh},
                              rsrc.memory_pool); // V 投影输出

  auto q_norm_buf = Tensor::buffer(dt_logits, {ntok, nh * dh},
                                   rsrc.memory_pool); // Q 归一化后
  auto k_norm_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh},
                                   rsrc.memory_pool); // K 归一化后
  auto gate_buf =
      Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool); // Gate 投影输出
  auto up_buf =
      Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool); // Up 投影输出
  auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
  auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
  auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
  auto result_cpu = std::vector<int64_t>(nreq);

  // Validate all buffer allocations
  std::vector<std::pair<std::shared_ptr<Tensor>, std::string>> buffers = {
      {q_buf, "q_buf"},           {k_buf, "k_buf"},
      {v_buf, "v_buf"},           {q_norm_buf, "q_norm_buf"},
      {k_norm_buf, "k_norm_buf"}, {gate_buf, "gate_buf"},
      {up_buf, "up_buf"},         {o_buf, "o_buf"},
      {prob_buf, "prob_buf"},     {result_buf, "result_buf"}};

  for (const auto &[buffer, name] : buffers) {
    if (!buffer) {
      throw std::runtime_error("Failed to allocate " + name +
                               " buffer - insufficient memory or pool issues");
    }
  }

  // if (g_debug_enabled && idev == 0) {
  //     printf("  Memory pool usage: %.2f MB total needed\n",
  //     total_memory_needed / (1024.0 * 1024.0));

  //     #ifdef _WIN32
  //     #else
  //     #endif
  // }

  // if (g_debug_enabled && idev == 0) {
  //     for (uint32_t i = 0; i < std::min(ntok, 10u); i++) {
  //         printf("%u ", tokens[i]);
  //     }
  //     printf("\n");
  //     printf("  meta.dt_logits: %d\n", meta.dt_logits);
  //     printf("  dt_logits (local): %d\n", dt_logits);
  //     printf("  INFINI_DTYPE_F16: %d\n", INFINI_DTYPE_F16);
  //     printf("  INFINI_DTYPE_F32: %d\n", INFINI_DTYPE_F32);
  //     printf("  Weight embedding dtype: %d\n", rsrc.w_in_embd->dtype());

  //     printf("  - Pointer: %p\n", rsrc.w_in_embd.get());
  //     printf("  - Data pointer: %p\n", rsrc.w_in_embd->data());
  //     printf("  - Dtype: %d\n", rsrc.w_in_embd->dtype());

  //     printf("  - Dtype meaning: ");
  //     switch(rsrc.w_in_embd->dtype()) {
  //         case INFINI_DTYPE_F16: printf("FLOAT16 (2 bytes)"); break;
  //         case INFINI_DTYPE_BF16: printf("BFLOAT16 (2 bytes)"); break;
  //         case INFINI_DTYPE_F32: printf("FLOAT32 (4 bytes)"); break;
  //         case INFINI_DTYPE_I32: printf("INT32 (4 bytes)"); break;
  //         case INFINI_DTYPE_U32: printf("UINT32 (4 bytes)"); break;
  //         default: printf("UNKNOWN(%d)", rsrc.w_in_embd->dtype()); break;
  //     }
  //     printf("\n");

  //     printf("  - Shape: ");
  //     auto in_shape = rsrc.w_in_embd->shape();
  //     for (size_t i = 0; i < in_shape.size(); i++) {
  //         printf("%zu%s", in_shape[i], i < in_shape.size()-1 ? "x" : "");
  //     }
  //     printf("\n");

  if (tokens[0] < meta.dvoc) {
    printf("DEBUG: Checking embedding for token %u:\n", tokens[0]);

    if (rsrc.w_in_embd->dtype() == INFINI_DTYPE_F16) {
      std::vector<uint16_t> temp_embedding_raw(std::min(size_t(10), d));
      size_t token_offset = tokens[0] * d;

      RUN_INFINI(infinirtMemcpy(
          temp_embedding_raw.data(), rsrc.w_in_embd->data(token_offset),
          sizeof(uint16_t) * temp_embedding_raw.size(), INFINIRT_MEMCPY_D2H));
      RUN_INFINI(infinirtDeviceSynchronize());

      printf("  First 10 embedding values (FP16 as hex): ");
      for (size_t i = 0; i < temp_embedding_raw.size(); i++) {
        printf("0x%04x ", temp_embedding_raw[i]);
      }
      printf("\n");

      // 如果有 FP16 到 float 的转换函数，也打印转换后的值
      printf("  First 10 embedding values (converted to float): ");
      for (size_t i = 0; i < temp_embedding_raw.size(); i++) {
        // 简单的 FP16 到 float 转换（需要实际的转换函数）
        // 这里只是示例，你需要使用正确的转换
        printf("%.6e ", 0.0f); // 占位符
      }
      printf("\n");
    } else {
      std::vector<float> temp_embedding(std::min(size_t(10), d));
      size_t token_offset = tokens[0] * d;

      RUN_INFINI(infinirtMemcpy(
          temp_embedding.data(), rsrc.w_in_embd->data(token_offset),
          sizeof(float) * temp_embedding.size(), INFINIRT_MEMCPY_D2H));
      RUN_INFINI(infinirtDeviceSynchronize());

      printf("  First 10 embedding values: ");
      for (size_t i = 0; i < temp_embedding.size(); i++) {
        printf("%.6e ", temp_embedding[i]);
      }
      printf("\n");
    }
  }
}

/*
 * 输入准备和 Token 嵌入查找
 */

// 准备输入
auto batch_pos_ids = std::vector<uint32_t>(ntok);
size_t req_start = 0;

// 构建位置 ID 数组：连接每个请求的位置序列
for (uint32_t req = 0; req < nreq; req++) {
  for (uint32_t i = 0; i < req_lens[req]; i++) {
    batch_pos_ids[req_start + i] = req_pos[req] + i;
  }
  req_start += req_lens[req];
}

// 将位置 ID 复制到设备内存（CPU 设备可以直接使用主机指针）
std::shared_ptr<Tensor> pos_ids_buf;
if (rsrc.device == INFINI_DEVICE_CPU) {
  pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
} else {
  pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
  // 位置 ID 的异步主机到设备复制 [ntok]
  RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(),
                                 sizeof(uint32_t) * ntok, INFINIRT_MEMCPY_H2D,
                                 stream));
}

// 查找输入嵌入：logits_in[i] = w_in_embd[tokens[i]]
for (uint32_t i = 0; i < ntok; i++) {
  if (tokens[i] >= meta.dvoc) {
    printf("Error: Invalid token ID %u >= vocab_size %zu\n", tokens[i],
           meta.dvoc);
    throw std::runtime_error("Invalid token ID");
  }
  // 添加详细的内存复制调试
  if (g_debug_enabled && idev == 0 && i == 0) {
    printf("DEBUG: Copying embedding for token %u to position %u\n", tokens[i],
           i);
    printf("  Source: %p (offset: %zu)\n", rsrc.w_in_embd->data(tokens[i] * d),
           tokens[i] * d);
    printf("  Dest: %p (offset: %zu)\n", logits_in->data(i * d), i * d);
    printf("  Size: %zu bytes\n", dsize(dt_logits) * d);
  }

  RUN_INFINI(infinirtMemcpyAsync(
      logits_in->data(i * d),              // 目标：logits_in 的第 i 行
      rsrc.w_in_embd->data(tokens[i] * d), // 源：tokens[i] 的嵌入
      dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
}

// 同步并检查结果
RUN_INFINI(infinirtStreamSynchronize(stream));

if (g_debug_enabled && idev == 0) {
  printf("DEBUG: After embedding lookup, checking first token result:\n");

  // 检查复制后的结果
  std::vector<uint16_t> result_check_raw(std::min(size_t(10), d));
  RUN_INFINI(infinirtMemcpy(result_check_raw.data(),
                            logits_in->data(0), // 第一个token的嵌入
                            sizeof(uint16_t) * result_check_raw.size(),
                            INFINIRT_MEMCPY_D2H));
  RUN_INFINI(infinirtDeviceSynchronize());

  printf("  First 10 values after copy: ");
  for (size_t i = 0; i < result_check_raw.size(); i++) {
    printf("0x%04x ", result_check_raw[i]);
  }
  printf("\n");

  printf("  Comparing with original embedding values: ");
  printf("Original: 0x2660 0x9da8 0xac50 0xa518 0xa8c0\n");
  printf("  Copied:   ");
  for (size_t i = 0; i < std::min(size_t(5), result_check_raw.size()); i++) {
    printf("0x%04x ", result_check_raw[i]);
  }
  printf("\n");
} else {
  std::vector<float> result_check(std::min(size_t(10), d));
  RUN_INFINI(infinirtMemcpy(result_check.data(), logits_in->data(0),
                            sizeof(float) * result_check.size(),
                            INFINIRT_MEMCPY_D2H));
  RUN_INFINI(infinirtDeviceSynchronize());

  printf("  First 10 values after copy: ");
  for (size_t i = 0; i < result_check.size(); i++) {
    printf("%.6e ", result_check[i]);
  }
  printf("\n");
}

// Debug: Save input embeddings
if (g_debug_enabled) {
  // Create input tokens tensor for debugging
  auto tokens_tensor =
      Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
  RUN_INFINI(infinirtMemcpy(tokens_tensor->data(), tokens,
                            sizeof(uint32_t) * ntok, INFINIRT_MEMCPY_H2D));
  save_tensor_debug_f32(tokens_tensor, "input_ids", -1, "cpp");

  // 根据实际数据类型调用正确的保存函数
  if (dt_logits == INFINI_DTYPE_F16) {
    save_tensor_debug_f16(logits_in, "input_embeddings", -1, "cpp");
  } else {
    save_tensor_debug_f32(logits_in, "input_embeddings", -1, "cpp");
  }
}

/*
 * InfiniCore 操作符描述符创建和工作区大小计算
 */

// 准备操作符和工作区
size_t workspace_size = 0, temp_size = 0;
infiniopRMSNormDescriptor_t desc_norm;

RUN_INFINI(infiniopCreateRMSNormDescriptor(
    rsrc.handle, &desc_norm, logits_in->desc(), // 输入：[ntok, d]
    logits_out->desc(),
    rsrc.w_attn_norm[0]->desc(), // 输出：[ntok, d]，权重：[d]
    meta.epsilon));              // 归一化 epsilon

RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm, &workspace_size));

workspace_size = std::max(workspace_size, temp_size);

/*
 * 注意力机制描述符
 *
 * 注意力计算涉及几个矩阵运算：
 * 1. QKV 投影：X -> Q、K、V 通过矩阵乘法
 * 2. Q 和 K 上的 RoPE 位置编码
 * 3. 注意力计算：softmax(QK^T/√d_k) * V
 * 4. 输出投影：O -> 最终注意力输出
 */

// QKV 投影和注意力输出的 GEMM 描述符
infiniopGemmDescriptor_t desc_attn_q, desc_attn_k, desc_attn_v, desc_attn_o;
/*
 * QKV 投影 GEMM：logits_in * w_attn_qkv -> qkv_buf
 *
 * 矩阵乘法：Y = X * W
 * 输入 X：[ntok, d] - 归一化的隐藏状态
 * 权重 W：[d, (nh + 2*nkvh)/ndev * dh] - QKV 投影权重
 * 输出 Y：[ntok, (nh + 2*nkvh)/ndev * dh] - 连接的 Q、K、V 投影
 *
 * 输出包含沿最后一个维度连接的 Q、K、V 投影：
 * - Q：[ntok, nh/ndev * dh]      （查询投影）
 * - K：[ntok, nkvh/ndev * dh]    （键投影）
 * - V：[ntok, nkvh/ndev * dh]    （值投影）
 */

RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_attn_q,
                                        q_buf->desc(), logits_out->desc(),
                                        rsrc.w_attn_q_proj[0]->desc()));
RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_q, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_attn_k,
                                        k_buf->desc(), logits_out->desc(),
                                        rsrc.w_attn_k_proj[0]->desc()));
RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_k, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_attn_v,
                                        v_buf->desc(), logits_out->desc(),
                                        rsrc.w_attn_v_proj[0]->desc()));
RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_v, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

// 重构多头 Q/K 归一化处理

/*
 * 多头 Q/K 归一化描述符创建
 *
 * Qwen3 的 Q/K 归一化是按头独立应用的：
 * - 每个头有独立的归一化权重 [dh]
 * - 需要为每个头创建单独的归一化描述符
 * - 或者使用循环在推理时分别处理每个头
 */

// 方案1：创建单个头的归一化描述符，在推理循环中重复使用
infiniopRMSNormDescriptor_t desc_q_norm_single, desc_k_norm_single;

// 创建单个头的张量描述符 [ntok, dh]
auto q_single_head = TensorDesc::create(dt_logits, {ntok, dh});
auto q_norm_single_head = TensorDesc::create(dt_logits, {ntok, dh});
auto k_single_head = TensorDesc::create(dt_logits, {ntok, dh});
auto k_norm_single_head = TensorDesc::create(dt_logits, {ntok, dh});

RUN_INFINI(infiniopCreateRMSNormDescriptor(
    rsrc.handle, &desc_q_norm_single, q_norm_single_head->desc(),
    q_single_head->desc(), rsrc.w_attn_q_norm[0]->desc(), meta.epsilon));
RUN_INFINI(infiniopCreateRMSNormDescriptor(
    rsrc.handle, &desc_k_norm_single, k_norm_single_head->desc(),
    k_single_head->desc(), rsrc.w_attn_k_norm[0]->desc(), meta.epsilon));

RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_q_norm_single, &temp_size));
workspace_size = std::max(workspace_size, temp_size);
RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_k_norm_single, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

/*
 * RoPE 描述符 - 仍然使用 3D 格式，但在推理时分头处理
 */
infiniopRoPEDescriptor_t desc_rope_q_single, desc_rope_k_single;

// 创建单个头的 3D RoPE 描述符 [ntok, 1, dh]
auto q_rope_single_head_3d = TensorDesc::create(dt_logits, {ntok, 1, dh});
auto k_rope_single_head_3d = TensorDesc::create(dt_logits, {ntok, 1, dh});

RUN_INFINI(infiniopCreateRoPEDescriptor(
    rsrc.handle, &desc_rope_q_single, q_rope_single_head_3d->desc(),
    q_rope_single_head_3d->desc(), pos_ids_buf->desc(), rsrc.sin_table->desc(),
    rsrc.cos_table->desc()));

RUN_INFINI(infiniopCreateRoPEDescriptor(
    rsrc.handle, &desc_rope_k_single, k_rope_single_head_3d->desc(),
    k_rope_single_head_3d->desc(), pos_ids_buf->desc(), rsrc.sin_table->desc(),
    rsrc.cos_table->desc()));

RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_q_single, &temp_size));
workspace_size = std::max(workspace_size, temp_size);
RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_k_single, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

// /*
// * RoPE 描述符 (更新为使用归一化后的张量)
// */
// // Q RoPE：q_norm_buf -> q_norm_buf (就地操作)

// auto q_norm_3d = q_norm_buf->dimSplit(1, {nh, dh});
// auto k_norm_3d = k_norm_buf->dimSplit(1, {nkvh, dh});

// printf("Debug: Creating Q RoPE descriptor\n");
// printf("Debug: Expected shapes - q_norm_3d: [%u, %zu, %zu]\n", ntok, nh, dh);

// RUN_INFINI(infiniopCreateRoPEDescriptor(
//     rsrc.handle, &desc_rope_q,
//     q_norm_3d->desc(), q_norm_3d->desc(),    // 输入/输出都是 3D
//     pos_ids_buf->desc(),                     // 位置 ID [ntok]
//     rsrc.sin_table->desc(),                  // 正弦表 [dctx, dh/2]
//     rsrc.cos_table->desc()));               // 余弦表 [dctx, dh/2]

// RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_q, &temp_size));
// workspace_size = std::max(workspace_size, temp_size);

// // K RoPE：k_norm_buf -> k_norm_buf (就地操作)
// RUN_INFINI(infiniopCreateRoPEDescriptor(
//     rsrc.handle, &desc_rope_k,
//     k_norm_3d->desc(), k_norm_3d->desc(),    // 输入/输出都是 3D
//     pos_ids_buf->desc(),                     // 位置 ID [ntok]
//     rsrc.sin_table->desc(),                  // 正弦表 [dctx, dh/2]
//     rsrc.cos_table->desc()));               // 余弦表 [dctx, dh/2]
// RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_k, &temp_size));
// workspace_size = std::max(workspace_size, temp_size);
// printf("Debug: RoPE descriptors created successfully\n");

/*
 * 注意力输出投影 GEMM：o_buf * w_attn_o_proj -> logits_in
 *
 * 矩阵乘法：Y = X * W
 * 输入 X：[ntok, nh/ndev * dh] - 此设备上所有头的注意力输出
 * 权重 W：[nh/ndev * dh, d] - 输出投影权重
 * 输出 Y：[ntok, d] - 投影的注意力输出（将在设备间累积）
 */
RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_attn_o,
                                        logits_in->desc(), o_buf->desc(),
                                        rsrc.w_attn_o_proj[0]->desc()));

// 计算 GEMM 操作的工作区需求
RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_o, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

/*
 * 每请求注意力内循环描述符
 *
 * 由于批处理中的每个请求可能有不同的序列长度和 KV 缓存状态，
 * 我们需要为每个请求的注意力计算单独的描述符。
 *
 * 每个请求的注意力机制包括：
 * 1. 分离的 QKV 投影：X -> Q、K、V (3个独立 GEMM)
 * 2. Q/K 归一化：Q -> Q_norm, K -> K_norm (Qwen3 特有)
 * 3. RoPE 应用到归一化后的 Q/K：Q_norm/K_norm -> 位置编码
 * 更新缓存（如果存在）
 * 4. 注意力计算：softmax(Q_norm * K_norm^T / √d_k) * V
 * 5. 输出投影：O -> 最终注意力输出
 *
 * 关键优化：
 * - 分组查询注意力：多个查询头可以共享 KV 头（ngroup = nh/nkvh）
 * - KV 缓存：存储和重用过去的键值对
 * - 因果掩码：未来 token 不能关注过去的 token
 */
// 注意力内层
auto desc_kv_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
auto desc_q_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);
auto desc_qk_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
auto desc_qk_softmaxs = std::vector<infiniopCausalSoftmaxDescriptor_t>(nreq);
auto desc_attn_v_gemms = std::vector<infiniopGemmDescriptor_t>(nreq);
auto desc_attn_v_rearranges = std::vector<infiniopRearrangeDescriptor_t>(nreq);

size_t token_offset = 0; // 跟踪当前请求在批处理中的位置
size_t max_qk_size = 0;  // 用于缓冲区分配的最大 QK 矩阵大小
size_t max_seq_len = 0;  // 用于缓冲区分配的最大序列长度

/*
 * 为每个请求的注意力计算创建描述符
 * 每个请求可能有不同的序列长度和过去的 KV 缓存长度，
 * 需要单独的描述符以获得最佳的内存布局和计算。
 */
for (uint32_t req = 0; req < nreq; req++) {
  auto past_len = req_pos[req];        // KV 缓存中已有的 token 数
  auto seq_len = req_lens[req];        // 要处理的当前序列长度
  auto total_len = past_len + seq_len; // KV 缓存中的总序列长度

  /*
   * 从批处理张量中提取每请求张量切片
   *
   * 此请求的张量形状：
   * - o：[seq_len, nh, dh] - 此请求的注意力输出
   * - q：[seq_len, nh, dh] - 此请求的查询向量
   * - k：[seq_len, nkvh, dh] - 此请求的键向量
   * - v：[seq_len, nkvh, dh] - 此请求的值向量（稍后使用）
   */

  // 步骤1：从2D缓冲区切片出当前请求的部分
  // 步骤1：从2D缓冲区切片出当前请求的部分
  auto o_2d = o_buf->slice({{0, token_offset, seq_len}}); // [seq_len, nh*dh]
  auto q_2d = q_norm_buf->slice(
      {{0, token_offset, seq_len}}); // [seq_len, nh*dh] - 使用归一化后的Q
  auto k_2d = k_norm_buf->slice(
      {{0, token_offset, seq_len}}); // [seq_len, nkvh*dh] - 使用归一化后的K

  // 步骤2：将2D切片分割成3D多头格式
  auto o = o_2d->dimSplit(1, {nh, dh}); // [seq_len, nh*dh] -> [seq_len, nh, dh]
  auto q = q_2d->dimSplit(
      1, {nh, dh}); // [seq_len, nh*dh] -> [seq_len, nh, dh] - 从归一化后的Q切片
  auto k = k_2d->dimSplit(1, {nkvh, dh});
  auto v = v_buf->slice({{0, token_offset, seq_len}}); // [seq_len, nkvh, dh]

  // auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh,
  // nkvh}});

  /*
   * KV 缓存张量配置
   * KV 缓存存储过去的键值对以实现高效的自回归生成。
   * 形状：[total_len, nkvh, dh] 在内存中存储为 [nkvh, dh, total_len]
   * full_kv：包括过去 + 当前 token 的完整 KV 缓存 [nkvh, dh, total_len]
   * cache_kv：用于存储当前键/值的切片 [nkvh, dh, seq_len]
   */
  // kv 缓存张量可以共享相同的描述符
  // [nkvh, dh, total_len]
  auto full_kv =
      kv_caches[req]->k[idev][0]->slice(0, 0, total_len)->permute({1, 2, 0});
  auto cache_kv = kv_caches[req]->k[idev][0]->slice(0, past_len, seq_len);
  /*
   * KV 重新排列描述符：将当前 K/V 存储在缓存中
   * 将当前键/值转换为缓存存储格式：
   * k：[seq_len, nkvh, dh] -> cache_kv：[seq_len, nkvh, dh]（不同的内存布局）
   */
  RUN_INFINI(infiniopCreateRearrangeDescriptor(
      rsrc.handle, &desc_kv_rearranges[req], cache_kv->desc(), k->desc()));

  /*
   * 分组查询注意力（GQA）的查询重新排列
   *
   * 重新塑造查询以实现高效的 GQA 计算：
   * q：[seq_len, nh, dh] -> [seq_len0, nkvh1, ngroup2, dh3] -> [nkvh1, ngroup2,
   * seq_len0, dh3] 此布局允许每个 KV 头关注多个查询头（每个 KV 头 ngroup
   * 个查询头）
   */
  // [nkvh, ngroup, seq_len, dh]
  q->dimSplit(1, {nkvh, ngroup})->permute({1, 2, 0, 3});
  auto q_t = TensorDesc::create(dt_logits, {nkvh, ngroup, seq_len, dh});
  // [seq_len, nkvh, ngroup, dh] -> [nkvh, ngroup, seq_len, dh]
  RUN_INFINI(infiniopCreateRearrangeDescriptor(
      rsrc.handle, &desc_q_rearranges[req], q_t->desc(), q->desc()));
  /*
   * 注意力值重新排列描述符
   * 在计算 attention_weights * values 后，重新排列回标准格式：
   * [nkvh, ngroup, seq_len, dh] -> [seq_len, nkvh, ngroup, dh] -> [seq_len, nh,
   * dh]
   */
  // [nkvh, ngroup, seq_len, dh] -> [seq_len, nkvh, ngroup, dh]
  auto attn_v_t = q_t;
  auto attn_v = TensorDesc::createWithOrder(
      dt_logits, {nkvh, ngroup, seq_len, dh}, {1, 2, 0, 3});
  RUN_INFINI(infiniopCreateRearrangeDescriptor(
      rsrc.handle, &desc_attn_v_rearranges[req], attn_v->desc(),
      attn_v_t->desc()));

  /*
   * QK 注意力分数计算：Q * K^T / √d_k
   *
   * 矩阵乘法计算注意力分数：
   * Q：[nkvh, ngroup * seq_len, dh]（为批处理计算重新塑造）
   * K^T：[nkvh, dh, total_len]（完整的 KV 缓存转置）
   * QK：[nkvh, ngroup * seq_len, total_len]（softmax 前的注意力分数）
   *
   * 缩放因子 1/√d_k 在 GEMM 操作期间应用。
   */
  q_t = TensorDesc::create(dt_logits, {nkvh, ngroup * seq_len, dh});
  auto qk = TensorDesc::create(dt_logits, {nkvh, ngroup * seq_len, total_len});
  max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
  max_seq_len = std::max(max_seq_len, size_t(seq_len));
  RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_qk_gemms[req],
                                          qk->desc(), q_t->desc(),
                                          full_kv->desc()));
  RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_qk_gemms[req], &temp_size));
  workspace_size = std::max(workspace_size, temp_size);

  /*
   * 注意力值计算：attention_weights * V
   *
   * 与注意力加权值的矩阵乘法：
   * attention_weights：[nkvh, ngroup * seq_len, total_len]（softmax 后）
   * V：[nkvh, total_len, dh]（完整的值缓存）
   * output：[nkvh, ngroup * seq_len, dh]（注意力输出）
   */
  // [nkvh, total_len, dh]
  auto full_v =
      kv_caches[req]->v[idev][0]->slice(0, 0, total_len)->permute({1, 0, 2});
  RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_attn_v_gemms[req],
                                          q_t->desc(), qk->desc(),
                                          full_v->desc()));
  RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_v_gemms[req], &temp_size));
  workspace_size = std::max(workspace_size, temp_size);

  /*
   * 带注意力掩码的因果 Softmax
   *
   * 将 softmax 应用于注意力分数并带有因果掩码（下三角）。
   * 形状：[nkvh * ngroup, seq_len, total_len]
   *
   * 因果掩码确保每个 token 只能关注之前的 token 和自身，
   * 防止在自回归生成期间未来 token 的信息泄露。
   */
  qk = TensorDesc::create(dt_logits, {nkvh * ngroup, seq_len, total_len});
  RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(
      rsrc.handle, &desc_qk_softmaxs[req], qk->desc(), qk->desc()));
  RUN_INFINI(
      infiniopGetCausalSoftmaxWorkspaceSize(desc_qk_softmaxs[req], &temp_size));
  workspace_size = std::max(workspace_size, temp_size);

  token_offset += seq_len;
}

/*
 * 分配注意力中间缓冲区
 *
 * 这些缓冲区存储注意力计算期间的中间结果。
 * 大小基于批处理中所有请求的最大需求。
 *
 * 缓冲区形状：
 * - qk_buf：[nh, max_qk_size] - 最大请求的注意力分数（QK^T）
 * - rearrange_q_buf：[nkvh, ngroup * max_seq_len, dh] - 重新排列的查询
 * - attn_val_buf：[nh, max_seq_len, dh] - 注意力输出值
 */
auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
auto rearrange_q_buf = Tensor::buffer(
    dt_logits, {nkvh, ngroup *max_seq_len, dh}, rsrc.memory_pool);
auto attn_val_buf =
    Tensor::buffer(dt_logits, {nh, max_seq_len, dh}, rsrc.memory_pool);

/*
 * 前馈网络（FFN）描述符
 *
 * FFN 块实现 SwiGLU 激活函数：
 * FFN(x) = (Swish(x * W_gate) ⊙ (x * W_up)) * W_down
 * 其中 Swish(x) = x * sigmoid(x) 且 ⊙ 是逐元素乘法
 *
 * 这涉及四个矩阵乘法：
 * 1. Gate 投影：X * W_gate -> gate_buf
 * 2. Up 投影：X * W_up -> up_buf
 * 3. SwiGLU 激活：gate_buf * swish(up_buf) -> gate_buf（复用）
 * 4. Down 投影：activated_output * W_down -> final_output
 */
// MLP 描述符
infiniopGemmDescriptor_t desc_mlp_gate, desc_mlp_up, desc_mlp_down;
infiniopSwiGLUDescriptor_t desc_swiglu;

/*
 * 分离的 MLP Gate 投影：logits_out [x] * w_mlp_gate_proj -> gate_buf
 *
 * 矩阵乘法：Y = X * W
 * 输入 X：[ntok, d] - 来自注意力的归一化隐藏状态
 * 权重 W：[d, di/ndev] - Gate 投影权重
 * 输出 Y：[ntok, di/ndev] - Gate 投影输出
 */
RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_mlp_gate,
                                        gate_buf->desc(), logits_out->desc(),
                                        rsrc.w_mlp_gate_proj[0]->desc()));
RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_mlp_gate, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

/*
 * 分离的 MLP Up 投影：logits_out * w_mlp_up -> up_buf
 *
 * 矩阵乘法：Y = X * W
 * 输入 X：[ntok, d] - 来自注意力的归一化隐藏状态
 * 权重 W：[d, di/ndev] - Up 投影权重
 * 输出 Y：[ntok, di/ndev] - Up 投影输出
 */
RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_mlp_up,
                                        up_buf->desc(), logits_out->desc(),
                                        rsrc.w_mlp_up_proj[0]->desc()));
RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_mlp_up, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

/*
 * MLP Down 投影：swiglu_output * w_mlp_down -> logits_in
 *
 * 矩阵乘法：Y = X * W
 * 输入 X：[ntok, di/ndev] - 来自 SwiGLU 的激活值
 * 权重 W：[di/ndev, d] - Down 投影权重
 * 输出 Y：[ntok, d] - 投影回模型维度
 */
RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_mlp_down,
                                        logits_in->desc(), gate_buf->desc(),
                                        rsrc.w_mlp_down_proj[0]->desc()));
RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_mlp_down, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

RUN_INFINI(
    infiniopCreateSwiGLUDescriptor(rsrc.handle, &desc_swiglu,
                                   gate_buf->desc(), // 输出缓冲区
                                   gate_buf->desc(), // gate 输入（复用缓冲区）
                                   up_buf->desc())); // up 输入
RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc_swiglu, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

/*
 * 输出生成和 Token 采样描述符
 *
 * 在所有 transformer 层之后，我们需要：
 * 1. 对每个请求的最后一个 token 应用最终层归一化
 * 2. 投影到词汇表空间以获得下一个 token 预测的 logits
 * 3. 应用采样（温度、top-k、top-p）来选择下一个 token
 */
// 输出和采样
infiniopRMSNormDescriptor_t desc_norm_out;

/*
 * 最终输出归一化
 *
 * 对每个请求的最后一个 token 的隐藏状态应用 RMSNorm。
 * 这在投影到词汇表之前归一化最终表示。
 *
 * 输入/输出形状：[1, d] -> [1, d]（一次处理一个请求）
 * 权重形状：[d] - 最终层归一化参数
 */
RUN_INFINI(infiniopCreateRMSNormDescriptor(
    rsrc.handle, &desc_norm_out, logits_out->slice(0, 0, 1)->desc(),
    logits_out->slice(0, 0, 1)->desc(), rsrc.w_out_norm->desc(), meta.epsilon));
RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_out, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

/*
 * 语言模型头投影：hidden_states -> vocabulary_logits
 *
 * 将归一化的隐藏状态投影到词汇表空间以获得下一个 token 预测的 logits。
 *
 * 矩阵乘法：Y = X * W
 * 输入 X：[nreq, d] - 每个请求的最终隐藏状态
 * 权重 W：[d, dvoc] - 语言模型头权重（通常与输入嵌入绑定）
 * 输出 Y：[nreq, dvoc] - 每个请求的词汇表上的 logits
 *
 * 这些 logits 代表每个可能的下一个 token 的未归一化对数概率。
 */
infiniopGemmDescriptor_t desc_out_embd;
RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_out_embd,
                                        prob_buf->desc(),
                                        logits_out->slice(0, 0, nreq)->desc(),
                                        rsrc.w_out_embd->desc()));
RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_out_embd, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

/*
 * 随机采样描述符
 *
 * 执行温度缩放、top-k 过滤、top-p（核）采样
 * 从概率分布中选择下一个 token。
 *
 * 采样过程：
 * 1. 应用温度缩放：logits = logits / temperature
 * 2. 应用 top-k 过滤：仅保留 k 个最高概率 token
 * 3. 应用 top-p 过滤：保留 token 直到累积概率 >= p
 * 4. 从过滤的分布中采样
 *
 * 输入：[dvoc] - 一个请求的词汇表上的 logits
 * 输出：标量 int64 - 选定的 token ID
 */
infiniopRandomSampleDescriptor_t desc_sample;
RUN_INFINI(infiniopCreateRandomSampleDescriptor(
    rsrc.handle, &desc_sample,
    TensorDesc::create(INFINI_DTYPE_I64, {}, {})->desc(), // 输出：标量 token ID
    TensorDesc::create(dt_logits, {dvoc}, {1})->desc())); // 输入：[dvoc] logits
RUN_INFINI(infiniopGetRandomSampleWorkspaceSize(desc_sample, &temp_size));
workspace_size = std::max(workspace_size, temp_size);

/*
 * 工作区分配
 *
 * 分配单个工作区缓冲区，可以处理所有操作的最大内存
 * 需求。这避免了推理期间频繁分配
 * 并确保高效的内存使用。
 */
// 分配工作区
std::shared_ptr<Storage> workspace_storage =
    Storage::createFromPool(workspace_size, rsrc.memory_pool);
void *workspace = workspace_storage->memory();

/*
 * ==================================================================================
 * 主 TRANSFORMER 推理计算循环
 * ==================================================================================
 *
 * 此部分执行通过所有 transformer 层的实际前向传递。
 * 每层包括：
 * 1. 带残差连接的多头注意力
 * 2. 带残差连接的前馈网络
 *
 * 计算遵循标准 transformer 架构：
 * x = x + Attention(LayerNorm(x))
 * x = x + FFN(LayerNorm(x))
 *
 * 对于分布式推理，注意力和 FFN 输出通过 all-reduce 操作
 * 在设备间累积。
 */

for (uint32_t layer = 0; layer < nlayer; layer++) {
  /*
   * ============================================================================
   * 多头注意力块
   * ============================================================================
   */

  // Debug: Save layer input (only for layer 0)
  if (g_debug_enabled && layer == 0) {
    save_tensor_debug_f16(logits_in, "input_hidden_states", layer);
  }

  // 1. 注意力

  /*
   * 注意力前层归一化
   *
   * 在注意力计算前对输入隐藏状态应用 RMSNorm。
   * 这遵循 "Pre-LN" transformer 架构以获得更好的训练稳定性。
   *
   * 公式：y = x / √(mean(x²) + ε) * γ
   * 输入：logits_in [ntok, d] - 来自前一层/嵌入的隐藏状态
   * 输出：logits_out [ntok, d] - 用于注意力的归一化隐藏状态
   * 权重：w_attn_norm[layer] [d] - 可学习的缩放参数
   */
  // rms 归一化
  RUN_INFINI(infiniopRMSNorm(desc_norm, workspace, workspace_size,
                             logits_out->data(), logits_in->data(),
                             rsrc.w_attn_norm[layer]->data(), stream));

  // Debug: Save attention norm output (only for layer 0)
  if (g_debug_enabled && layer == 0) {
    save_tensor_debug_f32(logits_out, "attn_norm_output", layer);
  }
  /*
   * Qwen3 注意力计算：分离 QKV + Q/K 归一化
   *
      同一输入分别乘三条权重 → 得到 Q、K、V
      reshape 成多头
      RoPE 只给 Q、K 加位置编码
      KV 复制到与 Q 头数一致
      softmax(QKᵀ)V → 合并 → 输出投影
   */

  // ============================================================================
  // 第一步：分离的 QKV 投影（替换原来的融合投影）
  // ============================================================================

  /*
   * Q 投影：logits_out * w_attn_q_proj -> q_buf
   */
  RUN_INFINI(infiniopGemm(desc_attn_q, workspace, workspace_size, q_buf->data(),
                          logits_out->data(), rsrc.w_attn_q_proj[layer]->data(),
                          1.0, 0.0, stream));

  /*
   * K 投影：logits_out * w_attn_k_proj -> k_buf
   */
  RUN_INFINI(infiniopGemm(desc_attn_k, workspace, workspace_size, k_buf->data(),
                          logits_out->data(), rsrc.w_attn_k_proj[layer]->data(),
                          1.0, 0.0, stream));

  /*
   * V 投影：logits_out * w_attn_v_proj -> v_buf
   */
  RUN_INFINI(infiniopGemm(desc_attn_v, workspace, workspace_size, v_buf->data(),
                          logits_out->data(), rsrc.w_attn_v_proj[layer]->data(),
                          1.0, 0.0, stream));

  // Debug: Save QKV projections (only for layer 0)
  if (g_debug_enabled && layer == 0) {
    save_tensor_debug_f32(q_buf, "attn_q_proj_raw", layer);
    save_tensor_debug_f32(k_buf, "attn_k_proj_raw", layer);
    save_tensor_debug_f32(v_buf, "attn_v_proj_raw", layer);

    // Validate QKV projections before normalization
    validate_tensor_range(q_buf, "q_buf before normalization");
    validate_tensor_range(k_buf, "k_buf before normalization");
  }

  // ============================================================================
  // 第二步：Qwen3 特有的 Q/K 归一化
  // ============================================================================

  /*
   * Q 归一化：q_buf -> q_norm_buf
   * 这是 Qwen3 特有的步骤，在 RoPE 之前对 Q 进行归一化
   */
  for (size_t head = 0; head < nh; head++) {
    // 计算当前头在缓冲区中的偏移（以 dh 为单位）
    size_t head_offset = head * dh;

    // 获取当前头的数据指针
    void *q_head_input = (char *)q_buf->data() + head_offset * dsize(dt_logits);
    void *q_head_output =
        (char *)q_norm_buf->data() + head_offset * dsize(dt_logits);

    // 对当前头应用归一化
    RUN_INFINI(infiniopRMSNorm(desc_q_norm_single, workspace, workspace_size,
                               q_head_output, // 输出：当前头的归一化结果
                               q_head_input, // 输入：当前头的原始数据
                               rsrc.w_attn_q_norm[layer]->data(), // 权重：[dh]
                               stream));
  }

  /*
   * K 归一化：k_buf -> k_norm_buf
   * 同样对 K 进行归一化
   */
  for (size_t head = 0; head < nkvh; head++) {

    // 计算当前头在缓冲区中的偏移
    size_t head_offset = head * dh;

    // 获取当前头的数据指针
    void *k_head_input = (char *)k_buf->data() + head_offset * dsize(dt_logits);
    void *k_head_output =
        (char *)k_norm_buf->data() + head_offset * dsize(dt_logits);

    // 对当前头应用归一化
    RUN_INFINI(infiniopRMSNorm(desc_k_norm_single, workspace, workspace_size,
                               k_head_output, // 输出：当前头的归一化结果
                               k_head_input, // 输入：当前头的原始数据
                               rsrc.w_attn_k_norm[layer]->data(), // 权重：[dh]
                               stream));
  }

  // Debug: Save Q/K normalized outputs (only for layer 0)
  if (g_debug_enabled && layer == 0) {
    // Validate normalized outputs
    validate_tensor_range(q_norm_buf, "attn_q_normed", -10.0, 10.0);
    validate_tensor_range(k_norm_buf, "attn_k_normed", -10.0, 10.0);

    // Clamp extreme values to prevent downstream issues
    clamp_tensor_inplace(q_norm_buf, -10.0f, 10.0f);
    clamp_tensor_inplace(k_norm_buf, -10.0f, 10.0f);

    save_tensor_debug_f32(q_norm_buf, "attn_q_normed", layer);
    save_tensor_debug_f32(k_norm_buf, "attn_k_normed", layer);
  }
  // ============================================================================
  // 第三步：按头应用 RoPE
  // ============================================================================

  for (size_t head = 0; head < nh; head++) {

    // 计算当前头在缓冲区中的偏移
    size_t head_offset = head * dh;

    // 获取当前头的数据指针（归一化后的）
    void *q_head_data =
        (char *)q_norm_buf->data() + head_offset * dsize(dt_logits);

    // 对当前头应用 RoPE（就地操作）
    RUN_INFINI(infiniopRoPE(desc_rope_q_single, workspace, workspace_size,
                            q_head_data, q_head_data, // 就地操作
                            pos_ids_buf->data(), rsrc.sin_table->data(),
                            rsrc.cos_table->data(), stream));
  }

  /*
   * 按头处理 K RoPE
   */
  for (size_t head = 0; head < nkvh; head++) {

    // 计算当前头在缓冲区中的偏移
    size_t head_offset = head * dh;

    // 获取当前头的数据指针（归一化后的）
    void *k_head_data =
        (char *)k_norm_buf->data() + head_offset * dsize(dt_logits);

    // 对当前头应用 RoPE（就地操作）
    RUN_INFINI(infiniopRoPE(desc_rope_k_single, workspace, workspace_size,
                            k_head_data, k_head_data, // 就地操作
                            pos_ids_buf->data(), rsrc.sin_table->data(),
                            rsrc.cos_table->data(), stream));
  }
  // ============================================================================
  // 第四步：每请求注意力计算（更新张量切片源）
  // ============================================================================

  size_t token_offset = 0;
  for (uint32_t req = 0; req < nreq; req++) {
    auto past_len = req_pos[req];
    auto seq_len = req_lens[req];

    /*
     * 从归一化后的缓冲区提取张量切片
     * 关键变化：使用 q_norm_buf, k_norm_buf, v_buf 而不是 qkv_buf 切片
     */
    auto o = o_buf->slice({{0, token_offset, seq_len}});
    auto q =
        q_norm_buf->slice({{0, token_offset, seq_len}}); // 使用归一化后的 Q
    auto k =
        k_norm_buf->slice({{0, token_offset, seq_len}}); // 使用归一化后的 K
    auto v = v_buf->slice({{0, token_offset, seq_len}}); // V 不需要额外归一化

    /*
     * 其余的注意力计算保持不变
     * 包括：KV 缓存更新、GQA 重排、QK 计算、softmax、注意力值计算
     */

    // KV 缓存更新（K 使用归一化后的值）
    RUN_INFINI(infiniopRearrange(
        desc_kv_rearranges[req],
        kv_caches[req]->k[idev][layer]->data(past_len * nkvh * dh), k->data(),
        stream)); // k 现在是归一化后的

    RUN_INFINI(infiniopRearrange(
        desc_kv_rearranges[req],
        kv_caches[req]->v[idev][layer]->data(past_len * nkvh * dh), v->data(),
        stream)); // v 保持不变

    // GQA 查询重排（Q 使用归一化后的值）
    RUN_INFINI(infiniopRearrange(desc_q_rearranges[req],
                                 rearrange_q_buf->data(), q->data(),
                                 stream)); // q 现在是归一化后的

    // QK 注意力分数计算
    RUN_INFINI(infiniopGemm(desc_qk_gemms[req], workspace, workspace_size,
                            qk_buf->data(), rearrange_q_buf->data(),
                            kv_caches[req]->k[idev][layer]->data(),
                            1. / sqrt(dh), 0.0, stream));

    // 因果 Softmax
    RUN_INFINI(infiniopCausalSoftmax(desc_qk_softmaxs[req], workspace,
                                     workspace_size, qk_buf->data(),
                                     qk_buf->data(), stream));

    // 注意力值计算
    RUN_INFINI(infiniopGemm(desc_attn_v_gemms[req], workspace, workspace_size,
                            attn_val_buf->data(), qk_buf->data(),
                            kv_caches[req]->v[idev][layer]->data(), 1.0, 0.0,
                            stream));

    // 输出重排
    RUN_INFINI(infiniopRearrange(desc_attn_v_rearranges[req], o->data(),
                                 attn_val_buf->data(), stream));

    token_offset += seq_len;
  }

  /*
   * 注意力输出投影和残差连接
   *
   * 将注意力输出投影回模型维度并添加残差连接。
   * 在分布式推理中，仅设备 0 添加残差连接以避免
   * 跨设备重复计算。
   *
   * 矩阵运算：Y = X * W + (如果 idev == 0 则残差 else 0)
   * 输入：o_buf [ntok, nh/ndev * dh] - 来自此设备的注意力输出
   * 权重：w_attn_out[layer] [nh/ndev * dh, d] - 输出投影权重
   * 输出：logits_in [ntok, d] - 带残差连接的投影输出
   */
  // o_proj
  RUN_INFINI(infiniopGemm(
      desc_attn_o, workspace, workspace_size, logits_in->data(), o_buf->data(),
      rsrc.w_attn_o_proj[layer]->data(), 1.0, idev == 0 ? 1.0 : 0.0, stream));
  // 残差：仅 rank 0 添加原始输入
  // 如果每个设备都添加完整的残差，最终结果会是：output = partial_outputs + ndev
  // * residual (错误)

  /*
   * 用于多设备推理的分布式 All-Reduce
   *
   * 在所有设备间求和注意力输出以完成分布式计算。
   * 每个设备计算了注意力头的一个切片，结果必须
   * 组合以获得完整的注意力输出。
   *
   * 操作：logits_in = sum(logits_in_device_i) 对于 i 在 [0, ndev) 范围内
   * 这同步所有设备并确保集群中的一致状态。
   */
  // 如果分布式则 All_reduce
  if (rsrc.comm != nullptr) {
    RUN_INFINI(infinicclAllReduce(logits_in->data(), logits_in->data(),
                                  ntok * d, dt_logits, INFINICCL_SUM, rsrc.comm,
                                  stream));
    RUN_INFINI(infinirtStreamSynchronize(stream)); // 通信后同步
  }

  // Debug: Save attention residual output (only for layer 0)
  if (g_debug_enabled && layer == 0) {
    save_tensor_debug_f32(logits_in, "attn_residual_output", layer);
  }

  /*
   * ============================================================================
   * 带 SwiGLU 激活的前馈网络（FFN）块
   * ============================================================================
   */

  /*
   * FFN 前层归一化
   *
   * 在 FFN 计算前对注意力输出应用 RMSNorm。
   *
   * 公式：y = x / √(mean(x²) + ε) * γ
   * 输入：logits_in [ntok, d] - 注意力输出 + 残差
   * 输出：logits_out [ntok, d] - 用于 FFN 处理的归一化
   * 权重：w_mlp_norm[layer] [d] - 可学习的缩放参数
   */
  // rms_norm
  RUN_INFINI(infiniopRMSNorm(desc_norm, workspace, workspace_size,
                             logits_out->data(), logits_in->data(),
                             rsrc.w_mlp_norm[layer]->data(), stream));

  // Debug: Save MLP norm output (only for layer 0)
  if (g_debug_enabled && layer == 0) {
    save_tensor_debug_f32(logits_out, "mlp_norm_output", layer);
  }

  /*
   * Qwen3 MLP 计算流程
   *
   * 计算顺序：
   * 1. gate = x @ W_gate               # Gate 投影（无激活）
   * 2. up = x @ W_up                   # Up 投影（无激活）
   * 3. intermediate = gate ⊙ SiLU(up)  # SwiGLU: gate * SiLU(up)
   * 4. output = intermediate @ W_down
   *
   * 对应 SwiGLU 公式：c_i = a_i ⊙ SiLU(b_i)
   * - a_i: gate (门控输入)
   * - b_i: up (要应用 SiLU 的输入)
   * - c_i: intermediate (输出结果)
   */

  // 步骤 1: Gate 投影（无激活函数）
  RUN_INFINI(infiniopGemm(desc_mlp_gate, workspace, workspace_size,
                          gate_buf->data(), logits_out->data(),
                          rsrc.w_mlp_gate_proj[layer]->data(), 1.0, 0.0,
                          stream));

  // 步骤 2: Up 投影（无激活函数）
  RUN_INFINI(infiniopGemm(desc_mlp_up, workspace, workspace_size,
                          up_buf->data(), logits_out->data(),
                          rsrc.w_mlp_up_proj[layer]->data(), 1.0, 0.0, stream));

  // Debug: Save MLP Gate and Up projections
  if (g_debug_enabled && layer == 0) {
    save_tensor_debug_f32(gate_buf, "mlp_gate_proj", layer);
    save_tensor_debug_f32(up_buf, "mlp_up_proj", layer);
  }

  // 步骤 3: SwiGLU 激活 - 一次性完成 gate ⊙ SiLU(up)
  RUN_INFINI(infiniopSwiGLU(
      desc_swiglu, workspace, workspace_size,
      gate_buf->data(), // 输出 (c): intermediate = gate ⊙ SiLU(up)
      up_buf->data(),   // 门控输入 (a): up_buf 投影结果
      gate_buf->data(), // SiLU输入 (b): gate_buf 投影结果，将应用 SiLU
      stream));

  // Debug: Save MLP intermediate (after SwiGLU) (only for layer 0)
  if (g_debug_enabled && layer == 0) {
    save_tensor_debug_f32(gate_buf, "mlp_intermediate", layer);
  }

  // 步骤 4: Down 投影和残差连接
  RUN_INFINI(infiniopGemm(desc_mlp_down, workspace, workspace_size,
                          logits_in->data(), gate_buf->data(),
                          rsrc.w_mlp_down_proj[layer]->data(), 1.0,
                          idev == 0 ? 1.0 : 0.0, stream));

  /*
   * FFN 下降投影和残差连接
   *
   * 将激活的 FFN 输出投影回模型维度并添加残差。
   * 与注意力一样，仅分布式推理中的设备 0 添加残差。
   *
   * 矩阵运算：Y = X * W + (如果 idev == 0 则残差 else 0)
   * 输入：gate_buf [ntok, di/ndev] - SwiGLU 激活值
   * 权重：w_ffn_down[layer] [di/ndev, d] - 下降投影权重
   * 输出：logits_in [ntok, d] - 带残差连接的 FFN 输出
   */
  RUN_INFINI(infiniopGemm(
      desc_mlp_down, workspace, workspace_size, logits_in->data(),
      gate_buf->data(), rsrc.w_mlp_down_proj[layer]->data(), 1.0,
      idev == 0 ? 1.0 : 0.0, stream)); // 残差：仅 rank 0 添加原始输入

  /*
   * FFN 输出的分布式 All-Reduce
   *
   * 在所有设备间求和 FFN 输出以完成分布式计算。
   * 每个设备计算了中间 FFN 维度的一个切片。
   */
  // 如果分布式则 All_reduce
  if (rsrc.comm != nullptr) {
    RUN_INFINI(infinicclAllReduce(logits_in->data(), logits_in->data(),
                                  ntok * d, dt_logits, INFINICCL_SUM, rsrc.comm,
                                  stream));
    RUN_INFINI(infinirtStreamSynchronize(stream)); // 通信后同步
  }

  // Debug: Save layer output (only for layer 0)
  if (g_debug_enabled && layer == 0) {
    save_tensor_debug_f32(logits_in, "layer_output", layer);
  }
}

/*
 * ==================================================================================
 * 输出生成和 TOKEN 采样
 * ==================================================================================
 *
 * 在处理完所有 transformer 层后，通过以下方式生成下一个 token：
 * 1. 对每个请求的最后一个 token 应用最终层归一化
 * 2. 投影到词汇表空间以获得 logits
 * 3. 使用温度、top-k 和 top-p 过滤采样下一个 token
 *
 * 仅设备 0 执行采样以避免重复计算。
 */
// 采样和输出
if (idev == 0) {
  /*
   * 每个请求的最终层归一化
   *
   * 对每个请求的最后一个 token 的隐藏状态应用 RMSNorm。
   * 最后一个 token 用于自回归生成中的下一个 token 预测。
   *
   * 对于每个请求，提取最后一个 token 的隐藏状态：
   * - token_offset 跟踪批处理中的累积位置
   * - 处理 req_lens[req] 个 token 后，最后一个 token 位于位置 (token_offset -
   * 1)
   */
  size_t token_offset = 0;
  for (uint32_t req = 0; req < nreq; req++) {
    auto seq_len = req_lens[req];
    token_offset += seq_len;

    /*
     * 为此请求归一化最后一个 token 的隐藏状态
     *
     * 输入：logits_in[(token_offset-1)*d : (token_offset)*d] - 最后一个 token
     * 的隐藏状态 [d] 输出：logits_out[req*d : (req+1)*d] -
     * 用于词汇表投影的归一化状态 [d] 权重：w_out_norm [d] - 最终层归一化参数
     */
    RUN_INFINI(infiniopRMSNorm(
        desc_norm_out, workspace, workspace_size,
        logits_out->data(req * d),               // 输出：请求 req 的 [d]
        logits_in->data((token_offset - 1) * d), // 输入：最后一个 token [d]
        rsrc.w_out_norm->data(), stream));
  }

  /*
   * 语言模型头投影
   *
   * 将归一化的最终隐藏状态投影到词汇表空间以获得 logits
   * 用于下一个 token 预测。
   *
   * 矩阵运算：logits = hidden_states * W_lm_head
   * 输入：logits_out [nreq, d] - 归一化的最终隐藏状态
   * 权重：w_out_embd [d, dvoc] - 语言模型头（通常与输入嵌入绑定）
   * 输出：prob_buf [nreq, dvoc] - 词汇表上的未归一化 logits
   */
  RUN_INFINI(infiniopGemm(desc_out_embd, workspace, workspace_size,
                          prob_buf->data(), logits_out->data(),
                          rsrc.w_out_embd->data(), 1.0, 0.0, stream));

  /*
   * 带温度和过滤的 Token 采样
   *
   * 对于每个请求，从概率分布中采样下一个 token
   * 使用温度缩放、top-k 过滤和 top-p（核）采样。
   *
   * 采样过程：
   * 1. 应用温度缩放：logits = logits / temperature
   * 2. 应用 top-k：仅保留 k 个最高概率 token
   * 3. 应用 top-p：保留 token 直到累积概率 >= p
   * 4. 使用随机值从过滤的分布中采样
   */
  std::random_device _rd;
  std::mt19937 gen(_rd());
  token_offset = 0;

  for (uint32_t req = 0; req < nreq; req++) {
    auto seq_len = req_lens[req];

    // 生成用于采样的随机值 [0, 1)
    float random_val = std::uniform_real_distribution<float>(0, 1)(gen);

    /*
     * 为此请求采样下一个 token
     *
     * 输入：prob_buf[req*dvoc : (req+1)*dvoc] - 词汇表上的 logits [dvoc]
     * 输出：result_buf[req] - 采样的 token ID
     * 参数：
     * - random_val：用于采样的随机种子
     * - topp[req]：核采样阈值（累积概率）
     * - topk[req]：top-k 过滤（保留前 k 个 token）
     * - temperature[req]：logits 的缩放因子（更高 = 更随机）
     */
    // prob_buf->debug();
    RUN_INFINI(infiniopRandomSample(
        desc_sample, workspace, workspace_size,
        result_buf->data(req),      // 输出：采样的 token ID
        prob_buf->data(req * dvoc), // 输入：此请求的 logits [dvoc]
        random_val,                 // 随机种子
        topp[req], topk[req], temperature[req], // 采样参数
        stream));
    // result_buf->debug();
    token_offset += seq_len;
  }

  /*
   * 将结果复制到主机内存
   *
   * 将采样的 token ID 从设备传输到主机内存以返回给调用者。
   * 同步流以确保所有计算在复制前完成。
   */
  RUN_INFINI(infinirtStreamSynchronize(stream));
  RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                            sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));

  // 将结果存储在输出数组中
  for (uint32_t req = 0; req < nreq; req++) {
    output[req] = result_cpu[req];
  }
}

/*
 * ==================================================================================
 * 描述符清理和资源释放
 * ==================================================================================
 *
 * 正确释放所有 InfiniCore 描述符以防止内存泄漏。
 * 描述符必须按依赖关系的相反顺序销毁。
 */
// 清理
infiniopDestroyRMSNormDescriptor(desc_norm);          // 层归一化
infiniopDestroyGemmDescriptor(desc_attn_q);           // 查询投影
infiniopDestroyGemmDescriptor(desc_attn_k);           // 键投影
infiniopDestroyGemmDescriptor(desc_attn_v);           // 值投影
infiniopDestroyGemmDescriptor(desc_attn_o);           // 注意力输出投影
infiniopDestroyRoPEDescriptor(desc_rope_q_single);    // 查询的 RoPE
infiniopDestroyRoPEDescriptor(desc_rope_k_single);    // 键的 RoPE
infiniopDestroyRMSNormDescriptor(desc_q_norm_single); // 单头查询归一化
infiniopDestroyRMSNormDescriptor(desc_k_norm_single); // 单头键归一化

// 清理每请求注意力描述符
for (uint32_t req = 0; req < nreq; req++) {
  infiniopDestroyRearrangeDescriptor(desc_kv_rearranges[req]); // KV 缓存存储
  infiniopDestroyRearrangeDescriptor(desc_q_rearranges[req]); // 查询重新排列
  infiniopDestroyGemmDescriptor(desc_qk_gemms[req]); // QK 注意力分数
  infiniopDestroyCausalSoftmaxDescriptor(desc_qk_softmaxs[req]); // 因果 softmax
  infiniopDestroyGemmDescriptor(desc_attn_v_gemms[req]); // 注意力值乘法
  infiniopDestroyRearrangeDescriptor(
      desc_attn_v_rearranges[req]); // 输出重新排列
}

// 清理 FFN 描述符
infiniopDestroyGemmDescriptor(desc_mlp_gate); // MLP Gate 投影
infiniopDestroyGemmDescriptor(desc_mlp_up);   // MLP Up 投影
infiniopDestroyGemmDescriptor(desc_mlp_down); // MLP Down 投影
infiniopDestroySwiGLUDescriptor(desc_swiglu); // SwiGLU 激活

// 清理输出描述符
infiniopDestroyRMSNormDescriptor(desc_norm_out);    // 最终层归一化
infiniopDestroyGemmDescriptor(desc_out_embd);       // 语言模型头
infiniopDestroyRandomSampleDescriptor(desc_sample); // Token 采样
}

/*
 * 批处理推理 API 函数（C 接口）
 *
 * 用于跨多个设备的分布式批处理推理的线程安全包装器。
 * 此函数使用条件变量同步的生产者-消费者模式
 * 协调模型中所有设备的推理。
 *
 * 参数：
 * - model：包含设备资源和工作线程的 JiugeModel 实例
 * - tokens：输入 token ID [ntok] - 来自所有请求的连接 token
 * - ntok：所有请求中的 token 总数
 * - req_lens：每个请求的长度 [nreq]
 * - nreq：批处理中的请求数
 * - req_pos：每个请求在 KV 缓存中的起始位置 [nreq]
 * - kv_caches：每个请求的 KV 缓存存储 [nreq]
 * - temperature/topk/topp：采样参数 [nreq]
 * - output：生成的 token ID [nreq] - 由此函数填充
 *
 * 线程同步：
 * 1. 主线程向所有工作线程发出开始推理信号
 * 2. 工作线程并行处理其分配的设备切片
 * 3. 主线程等待所有工作线程完成后再返回
 */
__C void inferQwen3Batch(struct Qwen3Model *model, const uint32_t *tokens,
                         uint32_t ntok, const uint32_t *req_lens, uint32_t nreq,
                         const uint32_t *req_pos,
                         struct Qwen3KVCache **kv_caches,
                         const float *temperature, const uint32_t *topk,
                         const float *topp, uint32_t *output) {
  /*
   * 将推理参数复制到模型的请求结构中
   * 这允许工作线程安全地访问请求数据。
   */
  model->req.tokens = tokens;
  model->req.ntok = ntok;
  model->req.req_lens = req_lens;
  model->req.nreq = nreq;
  model->req.req_pos = req_pos;
  model->req.kv_caches = kv_caches;
  model->req.output = output;
  model->req.temperature = temperature;
  model->req.topk = topk;
  model->req.topp = topp;

  /*
   * 向所有工作线程发出开始推理信号
   *
   * 每个设备都有一个在条件变量上等待的专用工作线程。
   * 设置 proceed=true 并通知唤醒工作线程以处理此批处理。
   */
  for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
    std::unique_lock<std::mutex> lock(model->states[idev].mtx);
    model->states[idev].proceed = true;
    lock.unlock();
    model->states[idev].cv_start.notify_one();
  }

  /*
   * 等待所有工作线程完成推理
   *
   * 以相反顺序等待以处理任何潜在的依赖关系。
   * 每个工作线程完成后将设置 proceed=false 并通知 cv_done。
   */
  for (size_t i = model->dev_ids.size(); i > 0; i--) {
    auto idev = i - 1;
    std::unique_lock<std::mutex> lock(model->states[idev].mtx);
    model->states[idev].cv_done.wait(
        lock, [&] { return !(model->states[idev].proceed); });
    lock.unlock();
  }
}

/*
 * 设备工作线程函数
 *
 * 每个设备在专用线程中运行此函数进行异步推理。
 * 线程生命周期：
 * 1. 初始化设备资源并发出就绪信号
 * 2. 在条件变量上等待推理请求
 * 3. 收到信号时执行设备特定的推理
 * 4. 发出完成信号并等待下一个请求
 * 5. 设置退出标志时清理资源
 *
 * 此设计支持高效的流水线并行和设备利用率。
 *
 * 参数：
 * - meta：模型架构元数据
 * - weights：模型权重张量
 * - rsrc：要填充的设备资源结构
 * - state：线程同步状态
 * - req：共享请求数据结构
 * - device：InfiniCore 设备类型
 * - idev/ndev：设备索引和设备总数
 * - dev_id：物理设备 ID
 * - comm：设备间通信上下文
 */
void launchDevice(const Qwen3Meta &meta, const Qwen3Weights *weights,
                  DeviceQwen3Resource *rsrc, InferState &state,
                  InferRequest &req, infiniDevice_t device, int idev, int ndev,
                  int dev_id, infinicclComm_t comm) {
  /*
   * 设备资源初始化
   *
   * 创建推理所需的所有设备特定资源。
   * 这包括权重、句柄、流和内存池。
   */
  // 创建设备资源
  createQwen3DeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id,
                            comm);

  /*
   * 发出设备就绪信号
   *
   * 通知主线程此设备已准备好进行推理。
   * 主线程等待所有设备加载完成后再继续。
   */
  {
    std::unique_lock<std::mutex> lock(state.mtx);
    state.loaded = true;
    lock.unlock();
    state.cv_load.notify_one();
  }

  /*
   * 主工作线程循环
   *
   * 等待推理请求并在退出请求时处理它们。
   * 这实现了一个生产者-消费者模式，其中主线程
   * 产生推理请求，工作线程消费它们。
   */
  // 推理循环
  while (true) {
    /*
     * 等待推理请求或退出信号
     *
     * 阻塞直到：
     * - proceed=true：新的推理请求可用
     * - exit_flag=true：请求关闭
     */
    std::unique_lock<std::mutex> lock(state.mtx);
    state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });

    // 如果请求关闭则优雅退出
    if (state.exit_flag) {
      break;
    }

    /*
     * 执行设备特定的推理
     *
     * 使用张量并行在此设备上处理当前批处理。
     * 函数处理此设备的计算切片。
     */
    inferQwen3DeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                          req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                          req.temperature, req.topk, req.topp, req.output);

    /*
     * 发出完成信号
     *
     * 标记此设备已完成并通知主线程。
     * 主线程等待所有设备返回结果。
     */
    state.proceed = false;
    lock.unlock();
    state.cv_done.notify_one();
  }

  /*
   * 资源清理
   *
   * 线程退出时释放所有设备资源。
   * 这确保在模型销毁期间正确清理。
   */
  // 清理
  releaseQwen3DeviceResource(*rsrc);
}

/*
 * JiugeModel 构造函数
 *
 * 初始化具有多个设备的分布式推理模型。
 * 设置工作线程、通信上下文和设备资源。
 *
 * 参数：
 * - _meta：模型架构元数据（层数、维度、数据类型）
 * - weights：模型权重张量
 * - device_：InfiniCore 设备类型（GPU/CPU）
 * - device_ids：用于分布式推理的物理设备 ID 列表
 *
 * 分布式设置：
 * - 为并行推理为每个设备创建一个工作线程
 * - 初始化 InfiniCCL 通信以实现多设备同步
 * - 等待所有设备完成初始化后再返回
 */
Qwen3Model::Qwen3Model(const Qwen3Meta *_meta, const Qwen3Weights *weights,
                       infiniDevice_t device_, std::vector<int> device_ids)
    : meta(*_meta) {
  int ndev = int(device_ids.size());
  device = device_;
  dev_ids = device_ids;
  dev_resources = std::vector<DeviceQwen3Resource>(ndev);
  states = std::vector<InferState>(ndev);
  threads.resize(ndev);

  /*
   * 初始化 InfiniCore 运行时
   *
   * 设置 InfiniCore 运行时环境以进行设备管理和
   * 操作执行。
   */
  RUN_INFINI(infinirtInit());

  /*
   * 初始化多设备通信
   *
   * 如果使用多个设备，则为分布式推理创建 InfiniCCL 通信器。
   * 通信支持设备间的同步和数据交换。
   */
  auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
  if (ndev > 1) {
    RUN_INFINI(
        infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
  }

  /*
   * 启动工作线程
   *
   * 为每个设备创建一个工作线程以处理异步推理。
   * 每个线程初始化其设备资源并等待推理请求。
   */
  for (int i = 0; i < ndev; i++) {
    threads[i] =
        std::thread(launchDevice, std::cref(meta), weights, &dev_resources[i],
                    std::ref(states[i]), std::ref(req), device, i, ndev,
                    dev_ids[i], comms[i]);
  }

  /*
   * 等待所有设备初始化
   *
   * 阻塞直到所有工作线程完成设备资源初始化。
   * 这确保在构造函数返回前模型完全就绪。
   */
  for (int i = 0; i < ndev; i++) {
    std::unique_lock<std::mutex> lock(states[i].mtx);
    states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
    lock.unlock();
  }
}

/*
 *
 *
 * 创建用于分布式推理的新实例。
 * 这是从 C/Python 代码创建模型的主要入口点。
 *
 * 参数：
 * - meta：模型架构元数据
 * - weights：模型权重张量
 * - device：InfiniCore 设备类型
 * - ndev：用于分布式推理的设备数
 * - dev_ids：物理设备 ID 数组 [ndev]
 *
 * 返回：指向新创建的实例的指针
 */
__C struct Qwen3Model *createQwen3Model(const Qwen3Meta *meta,
                                        const Qwen3Weights *weights,
                                        infiniDevice_t device, int ndev,
                                        const int *dev_ids) {
  // 将 C 数组转换为 C++ 向量以用于构造函数
  std::vector<int> device_ids(ndev);
  std::copy(dev_ids, dev_ids + ndev, device_ids.begin());

  // 创建并返回新模型实例
  Qwen3Model *model = new Qwen3Model(meta, weights, device, device_ids);
  return model;
}

__C void destroyQwen3Model(struct Qwen3Model *model) {
  auto ndev = model->dev_resources.size();

  /*
   * 向所有工作线程发出退出信号
   *
   * 为每个设备设置 exit_flag 并通知工作线程。
   * 这使它们脱离推理循环。
   */
  for (size_t idev = 0; idev < ndev; idev++) {
    std::unique_lock<std::mutex> lock(model->states[idev].mtx);
    model->states[idev].exit_flag = true;
    lock.unlock();
    model->states[idev].cv_start.notify_one();
  }

  /*
   * 等待所有线程终止
   *
   * 连接每个工作线程以确保干净关闭。
   * 这保证所有设备资源都得到正确释放。
   */
  for (size_t idev = 0; idev < ndev; idev++) {
    model->threads[idev].join();
  }

  // 释放模型实例
  delete model;
}

/*
 * 调试模式控制函数
 *
 * 允许从 Python 接口启用或禁用 C++ 调试输出。
 * 当启用时，会在推理过程中保存中间张量数据到文件。
 */
__C void setQwen3DebugMode(int enabled) {
  set_debug_mode(enabled != 0);
  if (enabled) {
    printf("Qwen3 debug mode enabled - will save intermediate tensors\n");
  } else {
    printf("Qwen3 debug mode disabled\n");
  }
}
