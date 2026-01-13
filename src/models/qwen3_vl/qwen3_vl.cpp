#include "qwen3_vl.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"

#include <cmath>
#include <random>
#include <thread>
#include <vector>

// 条件编译调试宏
#ifdef DEBUG_VISION
#define DEBUG_PRINT(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) \
    do {                      \
    } while (0)
#endif

// 常量定义
namespace Qwen3VLConstants {
constexpr uint32_t SPATIAL_MERGE_SIZE = 2;
constexpr uint32_t MERGE_UNIT = SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE; // 4
constexpr uint32_t IN_CHANNELS = 3;
constexpr uint32_t TEMPORAL_PATCH_SIZE = 2;
constexpr uint32_t PATCH_SIZE = 16;
constexpr uint32_t VISION_MLP_EXPANSION = 4; // vision_hidden_size * 4
constexpr uint32_t MAX_DEEPSTACK_LAYERS = 3;
constexpr uint32_t ROPE_SECTION_SIZE = 3;
constexpr uint32_t POS_IDS_2D_SIZE = 2;
constexpr uint32_t LLM_POS_IDS_3D_SIZE = 3;
// constexpr float EPSILON_DEFAULT = 1e-6f; // 暂时未使用，保留备用
} // namespace Qwen3VLConstants

inline void createDeviceResource(DeviceResource *rsrc, const Qwen3VLMeta *meta,
                                 std::shared_ptr<Qwen3VLDeviceWeight> weights,
                                 infiniDevice_t device, int idev,
                                 int ndev, int dev_id,
                                 infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    *rsrc = DeviceResource{
        device,
        dev_id,
        handle,
        weights,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

inline void releaseDeviceResource(DeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

std::tuple<std::shared_ptr<Tensor>, uint32_t> inferVision(const Qwen3VLMeta *meta, DeviceResource &rsrc,
                                                          const float *pixel_values, uint32_t num_patches,
                                                          const uint32_t *pos_ids, uint32_t pos_ids_len,
                                                          const uint32_t *llm_pos_ids, uint32_t llm_pos_ids_len,
                                                          const uint32_t *rope_section, uint32_t rope_section_len, int ndev) {
    auto d = meta->d;
    auto dt_logits = meta->dt_logits;
    auto stream = rsrc.stream;
    auto weight = rsrc.weights;

    // 若有视觉输入，先跑ViT得到visual_embeds；随后构建logits_in（视觉token用visual_embeds替换）
    std::shared_ptr<Tensor> vision_pos_ids_buf; // for ViT [patches, 2]
    std::shared_ptr<Tensor> llm_pos_ids_buf;    // for LLM [patches+text_len, 3]
    std::shared_ptr<Tensor> rope_section_buf;   // rope_section [3,]
    if (pos_ids != nullptr && pos_ids_len > 0) {
        assert(pos_ids_len % Qwen3VLConstants::POS_IDS_2D_SIZE == 0 && "pos_ids_len must be even for 2D mRoPE [patches, 2] format");
        assert(num_patches > 0 && "num_patches cannot be zero for 2D mRoPE");

        vision_pos_ids_buf = (rsrc.device == INFINI_DEVICE_CPU)
                               ? Tensor::weight(const_cast<uint32_t *>(pos_ids), INFINI_DTYPE_U32, {num_patches, 2})
                               : Tensor::buffer(INFINI_DTYPE_U32, {num_patches, 2}, rsrc.memory_pool);
        if (rsrc.device != INFINI_DEVICE_CPU) {
            RUN_INFINI(infinirtMemcpyAsync(vision_pos_ids_buf->data(), pos_ids, sizeof(uint32_t) * pos_ids_len,
                                           INFINIRT_MEMCPY_H2D, stream));
        }
    }

    // LLM 3D mRoPE参数处理：验证并构建llm_pos_ids和rope_section缓冲区
    if (llm_pos_ids != nullptr && llm_pos_ids_len > 0) {
        assert(llm_pos_ids_len % Qwen3VLConstants::LLM_POS_IDS_3D_SIZE == 0 && "llm_pos_ids_len must be divisible by 3 for 3D mRoPE [patches+text_len, 3] format");
        uint32_t total_tokens = llm_pos_ids_len / Qwen3VLConstants::LLM_POS_IDS_3D_SIZE;
        llm_pos_ids_buf = (rsrc.device == INFINI_DEVICE_CPU)
                            ? Tensor::weight(const_cast<uint32_t *>(llm_pos_ids), INFINI_DTYPE_U32, {total_tokens, 3})
                            : Tensor::buffer(INFINI_DTYPE_U32, {total_tokens, 3}, rsrc.memory_pool);
        if (rsrc.device != INFINI_DEVICE_CPU) {
            RUN_INFINI(infinirtMemcpyAsync(llm_pos_ids_buf->data(), llm_pos_ids, sizeof(uint32_t) * llm_pos_ids_len,
                                           INFINIRT_MEMCPY_H2D, stream));
        }
    }

    if (rope_section != nullptr && rope_section_len > 0) {
        assert(rope_section_len == Qwen3VLConstants::ROPE_SECTION_SIZE && "rope_section_len must be exactly 3 for [t, h, w] format");
        rope_section_buf = (rsrc.device == INFINI_DEVICE_CPU)
                             ? Tensor::weight(const_cast<uint32_t *>(rope_section), INFINI_DTYPE_U32, {3})
                             : Tensor::buffer(INFINI_DTYPE_U32, {Qwen3VLConstants::ROPE_SECTION_SIZE}, rsrc.memory_pool);
        if (rsrc.device != INFINI_DEVICE_CPU) {
            RUN_INFINI(infinirtMemcpyAsync(rope_section_buf->data(), rope_section, sizeof(uint32_t) * Qwen3VLConstants::ROPE_SECTION_SIZE,
                                           INFINIRT_MEMCPY_H2D, stream));
        }
    }

    std::shared_ptr<Tensor> visual_embeds; // [num_patches, vision_hidden_size]
    // Deepstack特征提取层索引
    // todo: 从config读取，默认[3,6,9]
    std::vector<uint32_t> deepstack_layers = {3, 6, 9};
    std::vector<std::shared_ptr<Tensor>> deepstack_features;

    DEBUG_PRINT("Vision processing: num_patches=%u", num_patches);
    // ===================1.patch_embd===================
    // todo py端读入进meta里
    // 根据权重形状确定实际参数: [vision_hidden_size, 3, temporal, patch, patch]
    uint32_t in_channels = Qwen3VLConstants::IN_CHANNELS;
    uint32_t temporal_patch_size = Qwen3VLConstants::TEMPORAL_PATCH_SIZE;
    uint32_t patch_size = Qwen3VLConstants::PATCH_SIZE;
    uint32_t vision_hidden_size = static_cast<uint32_t>(meta->vision_hidden_size);
    uint32_t patch_feature_dim = in_channels * temporal_patch_size * patch_size * patch_size;

    // 检查原始输入数据
    DEBUG_PRINT("=== 原始输入数据检查 ===");
    DEBUG_PRINT("num_patches=%u, in_channels=%u, temporal_patch_size=%u, patch_size=%u",
                num_patches, in_channels, temporal_patch_size, patch_size);
    DEBUG_PRINT("vision_hidden_size=%u, patch_feature_dim=%u", vision_hidden_size, patch_feature_dim);

    // // 按 num_patches 行, patch_feature_dim 列打印 pixel_values
    // DEBUG_PRINT("pixel_values 按 [num_patches, patch_feature_dim] 打印：");
    // for (uint32_t i = 0; i < num_patches; ++i) {
    //     std::string row_str = "pixel_values[" + std::to_string(i) + "]:";
    //     for (uint32_t j = 0; j < patch_feature_dim; ++j) {
    //         uint32_t idx = i * patch_feature_dim + j;
    //         char buf[64];
    //         snprintf(buf, sizeof(buf), " %f", pixel_values[idx]);
    //         row_str += buf;
    //     }
    //     printf("%s\n", row_str.c_str()); // 添加这行来实际打印
    // }

    // 输入像素: [num_patches, 3, 2, 16, 16]
    std::shared_ptr<Tensor> pixel_values_buf; // 外部传入的 pixel_values 是 float
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pixel_values_buf = Tensor::weight(const_cast<float *>(pixel_values), INFINI_DTYPE_F32,
                                          {num_patches, in_channels, temporal_patch_size, patch_size, patch_size});
    } else {
        pixel_values_buf = Tensor::buffer(INFINI_DTYPE_F32, {num_patches, in_channels, temporal_patch_size, patch_size, patch_size}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pixel_values_buf->data(), pixel_values,
                                       sizeof(float) * num_patches * patch_feature_dim,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    // conv buffer & config
    auto conv_output = Tensor::buffer(dt_logits, {num_patches, vision_hidden_size, 1, 1, 1}, rsrc.memory_pool);
    std::vector<int64_t> pads = {0, 0, 0};
    std::vector<int64_t> strides = {int64_t(temporal_patch_size), int64_t(patch_size), int64_t(patch_size)}; // strides = kernel_size
    std::vector<int64_t> dilations = {1, 1, 1};
    // 检查 conv3d 的输入数据
    // DEBUG_PRINT("=== conv3d 输入数据检查 ===");
    // DEBUG_PRINT("pixel_values_buf 信息:");
    // pixel_values_buf->debug();

    // patch_embed 权重
    // DEBUG_PRINT("weight->w_v_patch_embed_proj[0] 信息:");
    // weight->w_v_patch_embed_proj[0]->debug();
    // DEBUG_PRINT("weight->b_v_patch_embed_proj[0] 信息:");
    // weight->b_v_patch_embed_proj[0]->debug();

    // pos_embed 权重
    // DEBUG_PRINT("weight->w_v_pos_embed[0] 信息:");
    // weight->w_v_pos_embed[0]->debug();

    // merger 权重
    // DEBUG_PRINT("weight->w_v_merger_ln_q[0] 信息:");
    // weight->w_v_merger_ln_q[0]->debug();
    // DEBUG_PRINT("weight->b_v_merger_ln_q[0] 信息:");
    // weight->b_v_merger_ln_q[0]->debug();
    // DEBUG_PRINT("weight->w_v_merger_mlp_0[0] 信息:");
    // weight->w_v_merger_mlp_0[0]->debug();
    // DEBUG_PRINT("weight->b_v_merger_mlp_0[0] 信息:");
    // weight->b_v_merger_mlp_0[0]->debug();
    // DEBUG_PRINT("weight->w_v_merger_mlp_2[0] 信息:");
    // weight->w_v_merger_mlp_2[0]->debug();
    // DEBUG_PRINT("weight->b_v_merger_mlp_2[0] 信息:");
    // weight->b_v_merger_mlp_2[0]->debug();

    // // merger_list 权重 (只打印第一个)
    // DEBUG_PRINT("weight->w_v_merger_list_0_ln_q[0] 信息:");
    // weight->w_v_merger_list_0_ln_q[0]->debug();
    // DEBUG_PRINT("weight->b_v_merger_list_0_ln_q[0] 信息:");
    // weight->b_v_merger_list_0_ln_q[0]->debug();
    // DEBUG_PRINT("weight->w_v_merger_list_0_mlp_0[0] 信息:");
    // weight->w_v_merger_list_0_mlp_0[0]->debug();
    // DEBUG_PRINT("weight->b_v_merger_list_0_mlp_0[0] 信息:");
    // weight->b_v_merger_list_0_mlp_0[0]->debug();
    // DEBUG_PRINT("weight->w_v_merger_list_0_mlp_2[0] 信息:");
    // weight->w_v_merger_list_0_mlp_2[0]->debug();
    // DEBUG_PRINT("weight->b_v_merger_list_0_mlp_2[0] 信息:");
    // weight->b_v_merger_list_0_mlp_2[0]->debug();

    // // block0 权重和偏置
    // DEBUG_PRINT("weight->w_v_norm1[0] 信息:");
    // weight->w_v_norm1[0]->debug();
    // DEBUG_PRINT("weight->b_v_norm1[0] 信息:");
    // weight->b_v_norm1[0]->debug();
    // DEBUG_PRINT("weight->w_v_attn_proj[0] 信息:");
    // weight->w_v_attn_proj[0]->debug();
    // DEBUG_PRINT("weight->b_v_attn_proj[0] 信息:");
    // weight->b_v_attn_proj[0]->debug();
    DEBUG_PRINT("weight->w_v_attn_qkv[0] 信息:");
    weight->w_v_attn_qkv[0]->debug();
    DEBUG_PRINT("weight->b_v_attn_qkv[0] 信息:");
    weight->b_v_attn_qkv[0]->debug();
    // DEBUG_PRINT("weight->w_v_norm2[0] 信息:");
    // weight->w_v_norm2[0]->debug();
    // DEBUG_PRINT("weight->b_v_norm2[0] 信息:");
    // weight->b_v_norm2[0]->debug();
    // DEBUG_PRINT("weight->w_v_mlp_fc1[0] 信息:");
    // weight->w_v_mlp_fc1[0]->debug();
    // DEBUG_PRINT("weight->b_v_mlp_fc1[0] 信息:");
    // weight->b_v_mlp_fc1[0]->debug();
    // DEBUG_PRINT("weight->w_v_mlp_fc2[0] 信息:");
    // weight->w_v_mlp_fc2[0]->debug();
    // DEBUG_PRINT("weight->b_v_mlp_fc2[0] 信息:");
    // weight->b_v_mlp_fc2[0]->debug();

    DEBUG_PRINT("conv3d 参数: pads=[%ld,%ld,%ld], strides=[%ld,%ld,%ld], dilations=[%ld,%ld,%ld]",
                pads[0], pads[1], pads[2], strides[0], strides[1], strides[2],
                dilations[0], dilations[1], dilations[2]);

    DEBUG_PRINT("conv_output 形状: [%zu,%zu,%zu,%zu,%zu]",
                conv_output->shape()[0], conv_output->shape()[1], conv_output->shape()[2],
                conv_output->shape()[3], conv_output->shape()[4]);

    // // patch_embd
    // conv3d(conv_output,
    //        pixel_values_buf,
    //        weight->w_v_patch_embed_proj[0],
    //        weight->b_v_patch_embed_proj[0],
    //        pads, strides, dilations);

    // // 打印 conv3d 的结果
    // DEBUG_PRINT("=== conv3d 输出结果 ===");
    // conv_output->debug();

    exit(0);

    auto vit_hidden = conv_output->view({num_patches, vision_hidden_size});

    // ===================2.abs_pos_embd===================
    if (weight->w_v_pos_embed.size() > 0 && weight->w_v_pos_embed[0]) {
        // todo: 实现fast_pos_embed_interpolate的完整版本
        // 对于单图推理，我们使用线性插值来调整位置编码到当前图像尺寸
        auto pos_embed_weight = weight->w_v_pos_embed[0]; // [num_pos, vision_hidden_size]
        auto pos_embed_out = Tensor::buffer(dt_logits, {num_patches, vision_hidden_size}, rsrc.memory_pool);

        // 简化版本：直接取前num_patches个位置编码（假设位置编码表足够大）
        uint32_t available_pos = std::min(num_patches, static_cast<uint32_t>(pos_embed_weight->shape()[0]));
        if (available_pos > 0) {
            RUN_INFINI(infinirtMemcpyAsync(
                pos_embed_out->data(),
                pos_embed_weight->data(),
                dsize(dt_logits) * available_pos * vision_hidden_size,
                INFINIRT_MEMCPY_D2D, stream));

            // 如果num_patches > available_pos，用零填充剩余部分
            if (num_patches > available_pos) {
                auto zero_tensor = Tensor::buffer(dt_logits, {(num_patches - available_pos), vision_hidden_size}, rsrc.memory_pool);
                // 将零张量数据复制到位置编码输出的剩余部分
                RUN_INFINI(infinirtMemcpyAsync(
                    pos_embed_out->data(available_pos * vision_hidden_size),
                    zero_tensor->data(),
                    dsize(dt_logits) * (num_patches - available_pos) * vision_hidden_size,
                    INFINIRT_MEMCPY_D2D, stream));
            }

            // 添加位置编码到patch embeddings: vit_hidden = vit_hidden + pos_embeds
            add(vit_hidden, vit_hidden, pos_embed_out);
        }
    }

    // ===================3.vit_blocks===================
    uint32_t vision_layers = static_cast<uint32_t>(meta->vision_layers);
    uint32_t vision_heads = static_cast<uint32_t>(meta->vision_heads);
    uint32_t dh_v = vision_hidden_size / vision_heads;
    assert(dh_v * vision_heads == vision_hidden_size);

    DEBUG_PRINT("ViT configuration: layers=%u, heads=%u, hidden_size=%u, dh_v=%u",
                vision_layers, vision_heads, vision_hidden_size, dh_v);

    // 缓冲区
    auto vit_hidden_in = Tensor::buffer(dt_logits, {num_patches, vision_hidden_size}, rsrc.memory_pool);
    auto vit_hidden_out = Tensor::buffer(dt_logits, {num_patches, vision_hidden_size}, rsrc.memory_pool);
    vit_hidden_in->copyFrom(vit_hidden, rsrc.handle, stream);

    auto vit_qkv = Tensor::buffer(dt_logits, {num_patches, 3u * vision_hidden_size}, rsrc.memory_pool);
    auto vit_q = Tensor::buffer(dt_logits, {num_patches, vision_hidden_size}, rsrc.memory_pool);
    auto vit_k = Tensor::buffer(dt_logits, {num_patches, vision_hidden_size}, rsrc.memory_pool);
    auto vit_v = Tensor::buffer(dt_logits, {num_patches, vision_hidden_size}, rsrc.memory_pool);

    auto qk_v_buf = Tensor::buffer(dt_logits, {vision_heads, num_patches, num_patches}, rsrc.memory_pool);
    auto attn_val_v = Tensor::buffer(dt_logits, {vision_heads, num_patches, dh_v}, rsrc.memory_pool);

    // ===================3.1 attention(2d mRoPE)===================
    for (uint32_t vlayer = 0; vlayer < vision_layers; ++vlayer) {
        DEBUG_PRINT("ViT processing layer %u/%u", vlayer + 1, vision_layers);
        // ViT norm1: 在 [num_patches, 1, vision_hidden_size] 上做 LayerNorm
        {
            auto norm1_in_3d = vit_hidden_in->view({num_patches, 1u, vision_hidden_size});
            auto norm1_out_3d = Tensor::buffer(dt_logits, {num_patches, 1u, vision_hidden_size}, rsrc.memory_pool);
            auto norm1_input_standardization_3d = Tensor::buffer(dt_logits, {num_patches, 1u, vision_hidden_size}, rsrc.memory_pool);
            auto norm1_input_std_deviation_2d = Tensor::buffer(dt_logits, {num_patches, 1u}, rsrc.memory_pool);
            layernorm(norm1_out_3d,
                      norm1_input_standardization_3d,
                      norm1_input_std_deviation_2d,
                      norm1_in_3d,
                      weight->w_v_norm1[vlayer],
                      weight->b_v_norm1[vlayer],
                      meta->epsilon);
            rearrange(vit_hidden_out, norm1_out_3d->view({num_patches, vision_hidden_size}));
        }
        // QKV
        linear(vit_qkv, vit_hidden_out, weight->w_v_attn_qkv[vlayer], 1.0, 0.0, nullptr, weight->b_v_attn_qkv[vlayer]);
        // split q,k,v
        rearrange(vit_q, vit_qkv->slice(1, 0, vision_hidden_size));
        rearrange(vit_k, vit_qkv->slice(1, vision_hidden_size, vision_hidden_size));
        rearrange(vit_v, vit_qkv->slice(1, 2u * vision_hidden_size, vision_hidden_size));

        // 2D mRoPE on q,k: 使用 ViT 专用 (h,w) 位置编码
        assert(vision_pos_ids_buf != nullptr && "vision_pos_ids_buf cannot be nullptr");
        auto q_view = vit_q->view({vision_heads, num_patches, dh_v});
        auto k_view = vit_k->view({vision_heads, num_patches, dh_v});
        mrope_2d(q_view, q_view, vision_pos_ids_buf, weight->sin_table_v, weight->cos_table_v);
        mrope_2d(k_view, k_view, vision_pos_ids_buf, weight->sin_table_v, weight->cos_table_v);

        // Self-Attention: QK^T -> softmax -> *V
        {
            auto q_view = vit_q->view({vision_heads, num_patches, dh_v});
            auto k_view = vit_k->view({vision_heads, dh_v, num_patches});
            auto qk_view = qk_v_buf->view({vision_heads, num_patches, num_patches});
            linear(qk_view, q_view, k_view, 1.f / float(sqrt(dh_v)), 0.f, nullptr, nullptr);
            // ViT 使用非因果 softmax (无 mask)
            softmax(qk_view, qk_view);
            auto v_view = vit_v->view({vision_heads, num_patches, dh_v});
            linear(attn_val_v, qk_view, v_view, 1.f, 0.f, nullptr, nullptr);
        }

        // 合并 heads：[heads, num_patches, dh_v] -> [num_patches, heads*dh_v]
        auto attn_rearranged = Tensor::buffer(dt_logits, {num_patches, vision_hidden_size}, rsrc.memory_pool);
        auto attn_perm = attn_val_v->permute({1, 0, 2});
        // 变连续
        auto attn_contig = Tensor::buffer(dt_logits, {num_patches, vision_heads, dh_v}, rsrc.memory_pool);
        rearrange(attn_contig, attn_perm);
        // view
        auto attn_view = attn_contig->view({num_patches, vision_hidden_size});
        rearrange(attn_rearranged, attn_view);

        // out proj + 残差
        linear(vit_hidden_in, attn_rearranged, weight->w_v_attn_proj[vlayer], 1.0, 0.0, vit_hidden_in, weight->b_v_attn_proj[vlayer]);

        // ===================3.2 ffn===================
        // FFN 的 norm2: 在 [num_patches, 1, vision_hidden_size] 上做 LayerNorm
        {
            auto norm2_in_3d = vit_hidden_in->view({num_patches, 1u, vision_hidden_size});
            auto norm2_out_3d = Tensor::buffer(dt_logits, {num_patches, 1u, vision_hidden_size}, rsrc.memory_pool);
            auto norm2_input_standardization_3d = Tensor::buffer(dt_logits, {num_patches, 1u, vision_hidden_size}, rsrc.memory_pool);
            auto norm2_input_std_deviation_2d = Tensor::buffer(dt_logits, {num_patches, 1u}, rsrc.memory_pool);
            layernorm(norm2_out_3d,
                      norm2_input_standardization_3d,
                      norm2_input_std_deviation_2d,
                      norm2_in_3d,
                      weight->w_v_norm2[vlayer],
                      weight->b_v_norm2[vlayer],
                      meta->epsilon);
            rearrange(vit_hidden_out, norm2_out_3d->view({num_patches, vision_hidden_size}));
        }
        auto vit_fc1 = Tensor::buffer(dt_logits, {num_patches, Qwen3VLConstants::VISION_MLP_EXPANSION * vision_hidden_size}, rsrc.memory_pool);
        linear(vit_fc1, vit_hidden_out, weight->w_v_mlp_fc1[vlayer], 1.0, 0.0, nullptr, weight->b_v_mlp_fc1[vlayer]);
        auto vit_gelu = Tensor::buffer(dt_logits, {num_patches, Qwen3VLConstants::VISION_MLP_EXPANSION * vision_hidden_size}, rsrc.memory_pool);
        DEBUG_PRINT("ViT layer %u: applying GELU activation (should be gelu_pytorch_tanh)", vlayer);
        getInferenceContext().gelu(vit_gelu, vit_fc1);
        auto vit_fc2 = Tensor::buffer(dt_logits, {num_patches, vision_hidden_size}, rsrc.memory_pool);
        linear(vit_fc2, vit_gelu, weight->w_v_mlp_fc2[vlayer], 1.0, 0.0, nullptr, weight->b_v_mlp_fc2[vlayer]);
        // 残差
        rearrange(vit_hidden_in, vit_fc2->view({num_patches, vision_hidden_size}));

        // ===================3.3 deepstack_merger===================
        // 在指定层提取deepstack特征
        if (std::find(deepstack_layers.begin(), deepstack_layers.end(), vlayer) != deepstack_layers.end()) {
            size_t deepstack_idx = std::find(deepstack_layers.begin(), deepstack_layers.end(), vlayer) - deepstack_layers.begin();
            DEBUG_PRINT("Deepstack feature extraction at layer %u (deepstack_idx=%zu)", vlayer, deepstack_idx);

            if (deepstack_idx < Qwen3VLConstants::MAX_DEEPSTACK_LAYERS) { // 最多3个deepstack层
                const auto &w_ln_q = deepstack_idx == 0 ? weight->w_v_merger_list_0_ln_q
                                   : deepstack_idx == 1 ? weight->w_v_merger_list_1_ln_q
                                                        : weight->w_v_merger_list_2_ln_q;
                const auto &b_ln_q = deepstack_idx == 0 ? weight->b_v_merger_list_0_ln_q
                                   : deepstack_idx == 1 ? weight->b_v_merger_list_1_ln_q
                                                        : weight->b_v_merger_list_2_ln_q;
                const auto &w_mlp_0 = deepstack_idx == 0 ? weight->w_v_merger_list_0_mlp_0
                                    : deepstack_idx == 1 ? weight->w_v_merger_list_1_mlp_0
                                                         : weight->w_v_merger_list_2_mlp_0;
                const auto &b_mlp_0 = deepstack_idx == 0 ? weight->b_v_merger_list_0_mlp_0
                                    : deepstack_idx == 1 ? weight->b_v_merger_list_1_mlp_0
                                                         : weight->b_v_merger_list_2_mlp_0;
                const auto &w_mlp_2 = deepstack_idx == 0 ? weight->w_v_merger_list_0_mlp_2
                                    : deepstack_idx == 1 ? weight->w_v_merger_list_1_mlp_2
                                                         : weight->w_v_merger_list_2_mlp_2;
                const auto &b_mlp_2 = deepstack_idx == 0 ? weight->b_v_merger_list_0_mlp_2
                                    : deepstack_idx == 1 ? weight->b_v_merger_list_1_mlp_2
                                                         : weight->b_v_merger_list_2_mlp_2;

                // Deepstack merger：view->norm->MLP
                // use_postshuffle_norm=true: ln_q(x.view(-1, self.hidden_size))
                const uint32_t merge_unit = Qwen3VLConstants::MERGE_UNIT;
                uint32_t num_groups = num_patches / merge_unit;
                uint32_t hidden_size_merged = vision_hidden_size * merge_unit;
                assert(num_patches >= merge_unit && weight->w_v_merger_mlp_0.size() > 0);

                // view：四合一
                auto ds_input = vit_hidden_in->view({num_groups, hidden_size_merged});

                // LayerNorm：以 [batch, channel, feature] 形式调用， batch=num_groups, channel=1, feature=hidden_size_merged
                auto ds_input_3d = ds_input->view({num_groups, 1u, hidden_size_merged});
                auto ds_norm_3d = Tensor::buffer(dt_logits, {num_groups, 1u, hidden_size_merged}, rsrc.memory_pool);
                auto ds_input_standardization_3d = Tensor::buffer(dt_logits, {num_groups, 1u, hidden_size_merged}, rsrc.memory_pool);
                auto ds_input_std_deviation_2d = Tensor::buffer(dt_logits, {num_groups, 1u}, rsrc.memory_pool);
                layernorm(ds_norm_3d, ds_input_standardization_3d, ds_input_std_deviation_2d, ds_input_3d, w_ln_q[0], b_ln_q[0], meta->epsilon);
                auto ds_norm = ds_norm_3d->view({num_groups, hidden_size_merged});

                // MLP: fc1 -> GELU -> fc2
                auto ds_fc1 = Tensor::buffer(dt_logits, {num_groups, hidden_size_merged}, rsrc.memory_pool);
                linear(ds_fc1, ds_norm, w_mlp_0[0]->permute({1, 0}), 1.0, 0.0, nullptr, b_mlp_0[0]);

                auto ds_gelu = Tensor::buffer(dt_logits, {num_groups, hidden_size_merged}, rsrc.memory_pool);
                DEBUG_PRINT("Deepstack merger %zu: applying GELU activation", deepstack_idx);
                getInferenceContext().gelu(ds_gelu, ds_fc1);

                auto ds_out = Tensor::buffer(dt_logits, {num_groups, d}, rsrc.memory_pool);
                linear(ds_out, ds_gelu, w_mlp_2[0]->permute({1, 0}), 1.0, 0.0, nullptr, b_mlp_2[0]);
                deepstack_features.push_back(ds_out);
            }
        }
    }
    // vit 输出
    visual_embeds = vit_hidden_in;

    // ===================4.merger===================
    // 主 Merger: norm->view->MLP
    // use_postshuffle_norm=false: ln_q(x).view(-1, self.hidden_size)
    DEBUG_PRINT("Starting Vision Merger processing: deepstack_features.size()=%zu", deepstack_features.size());
    const uint32_t merge_unit = Qwen3VLConstants::MERGE_UNIT;
    uint32_t num_groups = num_patches / merge_unit;
    uint32_t hidden_size_merged = vision_hidden_size * merge_unit;
    assert(num_patches >= merge_unit && weight->w_v_merger_mlp_0.size() > 0);

    // LayerNorm：以 [batch, channel, feature] 形式调用， batch=num_patches, channel=1, feature=vision_hidden_size
    auto merger_ln_in_3d = visual_embeds->view({num_patches, 1u, vision_hidden_size});
    auto merger_ln_out_3d = Tensor::buffer(dt_logits, {num_patches, 1u, vision_hidden_size}, rsrc.memory_pool);
    auto merger_ln_standardization_3d = Tensor::buffer(dt_logits, {num_patches, 1u, vision_hidden_size}, rsrc.memory_pool);
    auto merger_ln_stddev_2d = Tensor::buffer(dt_logits, {num_patches, 1u}, rsrc.memory_pool);
    layernorm(merger_ln_out_3d,
              merger_ln_standardization_3d,
              merger_ln_stddev_2d,
              merger_ln_in_3d,
              weight->w_v_merger_ln_q[0],
              weight->b_v_merger_ln_q[0],
              meta->epsilon);

    // view：四合一
    auto merger_in = merger_ln_out_3d->view({num_groups, hidden_size_merged});

    // MLP: fc1 -> GELU -> fc2
    auto merger_fc1 = Tensor::buffer(dt_logits, {num_groups, hidden_size_merged}, rsrc.memory_pool);
    linear(merger_fc1, merger_in, weight->w_v_merger_mlp_0[0]->permute({1, 0}), 1.0, 0.0, nullptr, weight->b_v_merger_mlp_0[0]);

    auto merger_gelu = Tensor::buffer(dt_logits, {num_groups, hidden_size_merged}, rsrc.memory_pool);
    DEBUG_PRINT("Main merger: applying GELU activation");
    getInferenceContext().gelu(merger_gelu, merger_fc1);

    auto merger_out = Tensor::buffer(dt_logits, {num_groups, d}, rsrc.memory_pool);
    linear(merger_out, merger_gelu, weight->w_v_merger_mlp_2[0]->permute({1, 0}), 1.0, 0.0, nullptr, weight->b_v_merger_mlp_2[0]);

    // ===================4.1 merger concat===================
    // 主merger和deepstack特征连接: [main_features] + deepstack_features (在特征维度连接)
    assert(!deepstack_features.empty());
    uint32_t total_dim = d * (1 + deepstack_features.size());
    auto concat_embeds = Tensor::buffer(dt_logits, {num_groups, total_dim}, rsrc.memory_pool);

    // 复制主特征
    RUN_INFINI(infinirtMemcpyAsync(
        concat_embeds->data(),
        merger_out->data(),
        dsize(dt_logits) * num_groups * d, INFINIRT_MEMCPY_D2D, stream));

    // 复制deepstack特征（按照提取顺序）
    for (size_t i = 0; i < deepstack_features.size(); ++i) {
        RUN_INFINI(infinirtMemcpyAsync(
            concat_embeds->data((i + 1) * num_groups * d),
            deepstack_features[i]->data(),
            dsize(dt_logits) * num_groups * d, INFINIRT_MEMCPY_D2D, stream));
    }

    visual_embeds = concat_embeds;

    // 四合一后num_patches=num_groups
    num_patches = num_groups;

    return std::make_tuple(visual_embeds, num_groups);
}

void inferDeviceBatch(const Qwen3VLMeta *meta, DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      const uint32_t *pos_ids, uint32_t pos_ids_len,
                      const uint32_t *llm_pos_ids, uint32_t llm_pos_ids_len,
                      const uint32_t *rope_section, uint32_t rope_section_len,
                      const float *pixel_values, uint32_t /*is_vision_mode*/, // 视觉数据指针，是否视觉模式不再需要
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output, void *last_logits) {
    // DEBUG: 推理开始
    printf("[DEBUG] Qwen3VL inferDeviceBatch START: idev=%u, ntok=%u, nreq=%u, has_vision=%s\n",
           idev, ntok, nreq, (pixel_values != nullptr) ? "true" : "false");
    exit(0);
    auto nlayer = meta->nlayer;
    auto nkvh = meta->nkvh / ndev;
    auto nh = meta->nh / ndev;
    auto ngroup = nh / nkvh;
    // auto dctx = meta.dctx;
    auto dh = meta->dh;
    auto d = meta->d;
    auto dt_logits = meta->dt_logits;
    auto di = meta->di / ndev;
    auto dvoc = meta->dvoc;
    auto stream = rsrc.stream;
    auto weight = rsrc.weights;
    bool has_qkv_bias = meta->has_qkv_bias;

    // 判断是否为prefill阶段（有vision且是第一次前向）
    bool has_vision = (pixel_values != nullptr);
    bool is_prefill = has_vision && (req_pos[0] == 0);

    // 计算实际patches数量
    uint32_t num_patches = 0;
    if (pos_ids != nullptr && pos_ids_len > 0) {
        num_patches = pos_ids_len / Qwen3VLConstants::POS_IDS_2D_SIZE;
    }

    uint32_t llm_ntok = is_prefill ? (ntok - 1 + num_patches) : ntok; // prefill: 14 - 1 image token + 600 patches = 613
    // printf("[DEBUG] is_prefill=%s, ntok=%u, llm_ntok=%u, num_patches=%u\n",
    //    is_prefill ? "true" : "false", ntok, llm_ntok, num_patches);
    DEBUG_PRINT("is_prefill=%s, ntok=%u, llm_ntok=%u, num_patches=%u",
                is_prefill ? "true" : "false", ntok, llm_ntok, num_patches);

    std::shared_ptr<Tensor> vision_pos_ids_buf; // for ViT [patches, 2]
    std::shared_ptr<Tensor> llm_pos_ids_buf;    // for LLM [patches+text_len, 3] - TODO: wire from API
    std::shared_ptr<Tensor> rope_section_buf;   // rope_section [3,] - TODO: wire from API
    if (pos_ids != nullptr && pos_ids_len > 0) {
        assert(pos_ids_len % Qwen3VLConstants::POS_IDS_2D_SIZE == 0 && "pos_ids_len must be even for 2D mRoPE [patches, 2] format");
        assert(num_patches > 0 && "num_patches cannot be zero for 2D mRoPE");

        vision_pos_ids_buf = (rsrc.device == INFINI_DEVICE_CPU)
                               ? Tensor::weight(const_cast<uint32_t *>(pos_ids), INFINI_DTYPE_U32, {num_patches, 2})
                               : Tensor::buffer(INFINI_DTYPE_U32, {num_patches, 2}, rsrc.memory_pool);
        if (rsrc.device != INFINI_DEVICE_CPU) {
            RUN_INFINI(infinirtMemcpyAsync(vision_pos_ids_buf->data(), pos_ids, sizeof(uint32_t) * pos_ids_len,
                                           INFINIRT_MEMCPY_H2D, stream));
        }
    }

    // LLM 3D mRoPE参数处理：验证并构建llm_pos_ids和rope_section缓冲区
    if (llm_pos_ids != nullptr && llm_pos_ids_len > 0) {
        assert(llm_pos_ids_len % Qwen3VLConstants::LLM_POS_IDS_3D_SIZE == 0 && "llm_pos_ids_len must be divisible by 3 for 3D mRoPE [patches+text_len, 3] format");
        uint32_t total_tokens = llm_pos_ids_len / Qwen3VLConstants::LLM_POS_IDS_3D_SIZE;
        llm_pos_ids_buf = (rsrc.device == INFINI_DEVICE_CPU)
                            ? Tensor::weight(const_cast<uint32_t *>(llm_pos_ids), INFINI_DTYPE_U32, {total_tokens, 3})
                            : Tensor::buffer(INFINI_DTYPE_U32, {total_tokens, 3}, rsrc.memory_pool);
        if (rsrc.device != INFINI_DEVICE_CPU) {
            RUN_INFINI(infinirtMemcpyAsync(llm_pos_ids_buf->data(), llm_pos_ids, sizeof(uint32_t) * llm_pos_ids_len,
                                           INFINIRT_MEMCPY_H2D, stream));
        }
    }

    if (rope_section != nullptr && rope_section_len > 0) {
        assert(rope_section_len == Qwen3VLConstants::ROPE_SECTION_SIZE && "rope_section_len must be exactly 3 for [t, h, w] format");
        rope_section_buf = (rsrc.device == INFINI_DEVICE_CPU)
                             ? Tensor::weight(const_cast<uint32_t *>(rope_section), INFINI_DTYPE_U32, {3})
                             : Tensor::buffer(INFINI_DTYPE_U32, {Qwen3VLConstants::ROPE_SECTION_SIZE}, rsrc.memory_pool);
        if (rsrc.device != INFINI_DEVICE_CPU) {
            RUN_INFINI(infinirtMemcpyAsync(rope_section_buf->data(), rope_section, sizeof(uint32_t) * Qwen3VLConstants::ROPE_SECTION_SIZE,
                                           INFINIRT_MEMCPY_H2D, stream));
        }
    }

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {llm_ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {llm_ntok, d}, rsrc.memory_pool);
    auto q_buf = Tensor::buffer(dt_logits, {llm_ntok, nh * dh}, rsrc.memory_pool);
    auto k_buf = Tensor::buffer(dt_logits, {llm_ntok, nkvh * dh}, rsrc.memory_pool);
    auto v_buf = Tensor::buffer(dt_logits, {llm_ntok, nkvh * dh}, rsrc.memory_pool);

    auto gate_buf = Tensor::buffer(dt_logits, {llm_ntok, di}, rsrc.memory_pool);
    auto up_buf = Tensor::buffer(dt_logits, {llm_ntok, di}, rsrc.memory_pool);

    auto o_buf = Tensor::buffer(dt_logits, {llm_ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    // vision infer
    std::shared_ptr<Tensor> visual_embeds;
    if (has_vision && is_prefill) {
        auto [embeds, output_patches] = inferVision(meta, rsrc, pixel_values, num_patches, pos_ids, pos_ids_len, llm_pos_ids, llm_pos_ids_len, rope_section, rope_section_len, ndev);
        visual_embeds = embeds;
        num_patches = output_patches;
    } else {
        visual_embeds = nullptr;
        num_patches = 0;
    }

    // img_embd 和 text_embd 拼接构建 logits_in
    if (is_prefill) {
        // Prefill阶段：文本token查表，视觉token用 visual_embeds 顺序展开
        size_t vis_idx = 0;
        uint32_t out_idx = 0;
        for (uint32_t i = 0; i < ntok; i++) {
            const bool is_image_tok = (meta->image_token_id != 0 && tokens[i] == meta->image_token_id);
            const bool is_video_tok = (meta->video_token_id != 0 && tokens[i] == meta->video_token_id);
            if (has_vision && (is_image_tok || is_video_tok) && visual_embeds) {
                // 将一个vision token展开为num_patches个patch
                for (size_t patch_idx = 0; patch_idx < num_patches && vis_idx < num_patches; patch_idx++, vis_idx++, out_idx++) {
                    uint32_t copy_dim = std::min<uint32_t>(d, static_cast<uint32_t>(visual_embeds->shape()[1]));
                    if (copy_dim > 0) {
                        RUN_INFINI(infinirtMemcpyAsync(
                            logits_in->data(out_idx * d),
                            visual_embeds->data(vis_idx * visual_embeds->shape()[1]),
                            dsize(dt_logits) * copy_dim, INFINIRT_MEMCPY_D2D, stream));
                    }
                }
            } else {
                RUN_INFINI(infinirtMemcpyAsync(
                    logits_in->data(out_idx * d),
                    weight->w_in_embd->data(tokens[i] * d),
                    dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
                out_idx++;
            }
        }
    } else {
        // Decode阶段：直接使用text token查表
        for (uint32_t i = 0; i < ntok; i++) {
            RUN_INFINI(infinirtMemcpyAsync(
                logits_in->data(i * d),
                weight->w_in_embd->data(tokens[i] * d),
                dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
        }
    }

    // Attention
    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }

    auto qk_buf = Tensor::buffer(dt_logits, {nh * max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    // Compute llm
    DEBUG_PRINT("Starting LLM processing: %zu layers", (size_t)nlayer);
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        DEBUG_PRINT("LLM processing layer %u/%zu", layer + 1, (size_t)nlayer);
        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, weight->w_attn_norm[layer], meta->epsilon);
        // qkv_proj
        linear(q_buf, logits_out,
               weight->w_attn_q[layer],
               1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_q[layer] : nullptr);
        linear(k_buf, logits_out,
               weight->w_attn_k[layer],
               1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_k[layer] : nullptr);
        // q/k-norm
        if (weight->w_q_norm.size() > layer && weight->w_k_norm.size() > layer) {
            auto q_norm_buf = Tensor::buffer(dt_logits, {llm_ntok, nh * dh}, rsrc.memory_pool);
            auto k_norm_buf = Tensor::buffer(dt_logits, {llm_ntok, nkvh * dh}, rsrc.memory_pool);
            rmsnorm(q_norm_buf, q_buf, weight->w_q_norm[layer], meta->epsilon);
            rmsnorm(k_norm_buf, k_buf, weight->w_k_norm[layer], meta->epsilon);
            rearrange(q_buf, q_norm_buf);
            rearrange(k_buf, k_norm_buf);
        }
        linear(v_buf, logits_out,
               weight->w_attn_v[layer],
               1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_v[layer] : nullptr);
        // RoPE处理：prefill阶段使用3D MRoPE，decode阶段使用普通RoPE
        if (is_prefill && llm_pos_ids_buf && rope_section_buf) {
            // Prefill阶段：3D MRoPE
            // printf("[DEBUG] MRoPE3D参数检查:\n");
            auto q_view = q_buf->view({nh, llm_ntok, dh});
            auto k_view = k_buf->view({nkvh, llm_ntok, dh});
            // printf("[DEBUG] q维度: [%zu, %zu, %zu] (nhead=%zu, llm_seqlen=%u, dhead=%zu)\n",
            //        q_view->shape()[0], q_view->shape()[1], q_view->shape()[2], nh, llm_ntok, dh);
            // printf("[DEBUG] k维度: [%zu, %zu, %zu] (nkvh=%zu, llm_seqlen=%u, dhead=%zu)\n",
            //        k_view->shape()[0], k_view->shape()[1], k_view->shape()[2], nkvh, llm_ntok, dh);
            // printf("[DEBUG] pos维度: [%zu, %zu]\n", llm_pos_ids_buf->shape()[0], llm_pos_ids_buf->shape()[1]);
            // printf("[DEBUG] sin维度: [%zu, %zu]\n", weight->sin_table->shape()[0], weight->sin_table->shape()[1]);
            // printf("[DEBUG] cos维度: [%zu, %zu]\n", weight->cos_table->shape()[0], weight->cos_table->shape()[1]);
            // printf("[DEBUG] rope_section维度: [%zu]\n", rope_section_buf->shape()[0]);

            mrope_3d(q_view, q_view, llm_pos_ids_buf, weight->sin_table, weight->cos_table, rope_section_buf);
            mrope_3d(k_view, k_view, llm_pos_ids_buf, weight->sin_table, weight->cos_table, rope_section_buf);
        } else if (!is_prefill) {
            // Decode阶段：普通RoPE，使用当前位置
            // printf("[DEBUG] 使用普通RoPE for decode阶段\n");
            auto pos_buf = Tensor::buffer(INFINI_DTYPE_U32, {llm_ntok}, rsrc.memory_pool);
            // decode时位置来自req_pos + current_step
            uint32_t current_pos = req_pos[0] + req_lens[0] - 1; // 当前生成位置
            uint32_t pos_data = current_pos;
            RUN_INFINI(infinirtMemcpyAsync(pos_buf->data(), &pos_data, sizeof(uint32_t), INFINIRT_MEMCPY_H2D, stream));

            // 添加decode阶段RoPE调试信息
            // printf("[DEBUG] Decode RoPE 参数调试:\n");
            // printf("[DEBUG] req_pos[0]=%u, req_lens[0]=%u, current_pos=%u\n", req_pos[0], req_lens[0], current_pos);
            // printf("[DEBUG] nh=%zu, nkvh=%zu, llm_ntok=%u, dh=%zu\n", nh, nkvh, llm_ntok, dh);

            auto q_view = q_buf->view({llm_ntok, nh, dh});
            auto k_view = k_buf->view({llm_ntok, nkvh, dh});
            // printf("[DEBUG] q_view维度: [%zu, %zu, %zu]\n", q_view->shape()[0], q_view->shape()[1], q_view->shape()[2]);
            // printf("[DEBUG] k_view维度: [%zu, %zu, %zu]\n", k_view->shape()[0], k_view->shape()[1], k_view->shape()[2]);
            // printf("[DEBUG] pos_buf维度: [%zu]\n", pos_buf->shape()[0]);
            // printf("[DEBUG] sin_table维度: [%zu, %zu]\n", weight->sin_table->shape()[0], weight->sin_table->shape()[1]);
            // printf("[DEBUG] cos_table维度: [%zu, %zu]\n", weight->cos_table->shape()[0], weight->cos_table->shape()[1]);

            // // RoPE维度一致性检查
            // printf("[DEBUG] RoPE维度检查:\n");
            // printf("[DEBUG] seqlen(llm_ntok)=%u, nhead(nh)=%zu, dhead(dh)=%zu\n", llm_ntok, nh, dh);
            // printf("[DEBUG] table_len(sin[0])=%zu, table_dim(sin[1])=%zu\n", weight->sin_table->shape()[0], weight->sin_table->shape()[1]);
            // printf("[DEBUG] pos_seqlen=%zu\n", pos_buf->shape()[0]);

            rope(q_view, q_view, pos_buf, weight->sin_table, weight->cos_table);
            rope(k_view, k_view, pos_buf, weight->sin_table, weight->cos_table);
        }

        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto total_len = past_len + seq_len;
            // printf("[DEBUG] KV Cache - req=%u: past_len=%u, seq_len=%u, total_len=%u\n", req, past_len, seq_len, total_len);
            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = q_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = k_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});
            auto v = v_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});

            // self attention
            // concat
            rearrange(kv_caches[req]->k[idev][layer]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][layer]->slice(0, past_len, seq_len), v);
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = kv_caches[req]->k[idev][layer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);
            auto v_gemm = kv_caches[req]->v[idev][layer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

            token_offset += seq_len;
        }

        // o_proj
        linear(logits_in, o_buf, weight->w_attn_out[layer],
               1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), llm_ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // 2. FFN
        rmsnorm(logits_out, logits_in, weight->w_ffn_norm[layer], meta->epsilon);
        linear(gate_buf, logits_out,
               weight->w_ffn_gate[layer],
               1.0, 0.0, nullptr, nullptr);
        linear(up_buf, logits_out,
               weight->w_ffn_up[layer],
               1.0, 0.0, nullptr, nullptr);
        DEBUG_PRINT("LLM layer %u: applying SwiGLU activation (SiLU-based)", layer);
        swiglu(gate_buf, up_buf, gate_buf);
        linear(logits_in, gate_buf,
               weight->w_ffn_down[layer],
               1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), llm_ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }

    // Sample and Output
    if (idev == 0) {
        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, weight->w_out_norm, meta->epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {llm_ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, weight->w_out_embd, 1.0, 0.0, nullptr, nullptr);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, last_logits_buf->data(), dsize(dt_logits) * llm_ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        weight->w_out_norm,
                        meta->epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), weight->w_out_embd, 1.0, 0.0, nullptr, nullptr);

            // [DEBUG]：在采样前检查prob_buf是否存在NaN/Inf，并统计范围
            RUN_INFINI(infinirtStreamSynchronize(stream));
            auto prob_cpu = std::vector<float>(nreq * dvoc);
            RUN_INFINI(infinirtMemcpy(prob_cpu.data(), prob_buf->data(), sizeof(float) * nreq * dvoc, INFINIRT_MEMCPY_D2H));
            size_t nan_count = 0, inf_count = 0;
            float global_min = std::numeric_limits<float>::infinity();
            float global_max = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < (size_t)nreq * (size_t)dvoc; ++i) {
                float v = prob_cpu[i];
                if (!std::isfinite(v)) {
                    if (std::isnan(v)) {
                        nan_count++;
                    } else {
                        inf_count++;
                    }
                } else {
                    if (v < global_min) {
                        global_min = v;
                    }
                    if (v > global_max) {
                        global_max = v;
                    }
                }
            }
            DEBUG_PRINT("prob_buf stats: nan=%zu, inf=%zu, min=%g, max=%g", nan_count, inf_count, global_min, global_max);
            (void)nan_count;
            (void)inf_count; // suppress unused variable warnings

            std::random_device _rd;
            std::mt19937 gen(_rd());
            token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
                randomSample(result_buf->slice(0, req, 1)->view_as({}, {}),
                             prob_buf->slice(0, req, 1)->view_as({dvoc}, {1}),
                             random_val, topp[req], topk[req], temperature[req]);
                token_offset += seq_len;
            }
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                      sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
            for (uint32_t req = 0; req < nreq; req++) {
                output[req] = uint32_t(result_cpu[req]);
            }
        }
    }

    DEBUG_PRINT("Qwen3VL inferDeviceBatch COMPLETED successfully");
}

__C void
inferBatchQwen3VL(struct Qwen3VLModel *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  const uint32_t *pos_ids, uint32_t pos_ids_len,
                  const uint32_t *llm_pos_ids, uint32_t llm_pos_ids_len,
                  const uint32_t *rope_section, uint32_t rope_section_len,
                  const float *pixel_values,
                  struct KVCache **kv_caches,
                  const float *temperature, const uint32_t *topk, const float *topp,
                  uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.pos_ids = pos_ids;
    model->req.pos_ids_len = pos_ids_len;
    model->req.llm_pos_ids = llm_pos_ids;
    model->req.llm_pos_ids_len = llm_pos_ids_len;
    model->req.rope_section = rope_section;
    model->req.rope_section_len = rope_section_len;
    model->req.pixel_values = pixel_values;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

__C void
forwardBatchQwen3VL(struct Qwen3VLModel *model,
                    const uint32_t *tokens, uint32_t ntok,
                    const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                    const uint32_t *pos_ids, uint32_t pos_ids_len,
                    const uint32_t *llm_pos_ids, uint32_t llm_pos_ids_len,
                    const uint32_t *rope_section, uint32_t rope_section_len,
                    const float *pixel_values,
                    struct KVCache **kv_caches,
                    void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.pos_ids = pos_ids;
    model->req.pos_ids_len = pos_ids_len;
    model->req.llm_pos_ids = llm_pos_ids;
    model->req.llm_pos_ids_len = llm_pos_ids_len;
    model->req.rope_section = rope_section;
    model->req.rope_section_len = rope_section_len;
    model->req.pixel_values = pixel_values;
    model->req.kv_caches = kv_caches;
    model->req.output = nullptr;
    model->req.logits = logits;
    model->req.temperature = nullptr;
    model->req.topk = nullptr;
    model->req.topp = nullptr;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

void launchDevice(const Qwen3VLMeta *meta, std::shared_ptr<Qwen3VLDeviceWeight> weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // Create Device Resource
    createDeviceResource(rsrc, meta, weights, device, idev, ndev, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }

        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.pos_ids, req.pos_ids_len,
                         req.llm_pos_ids, req.llm_pos_ids_len, req.rope_section, req.rope_section_len,
                         req.pixel_values, 0,
                         req.kv_caches, req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

Qwen3VLModel::Qwen3VLModel(const Qwen3VLMeta *meta, const ModelWeights *weights_) {
    auto weights = (Qwen3VLWeights *)(weights_);
    device = weights->device();
    dev_ids = weights->devIds();
    int ndev = int(dev_ids.size());
    dev_resources = std::vector<DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);

    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, meta, weights->device_weights()[i], &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct Qwen3VLModel *
createQwen3VLModel(const Qwen3VLMeta *meta,
                   const ModelWeights *weights) {
    Qwen3VLModel *model = new Qwen3VLModel(meta, weights);
    return model;
}

__C void destroyQwen3VLModel(struct Qwen3VLModel *model) {
    auto ndev = model->dev_resources.size();

    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}
