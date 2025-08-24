/*
 * Jiuge 模型权重提取工具
 * 
 * 此头文件提供从全局权重存储中提取和重塑模型权重到设备特定张量对象的实用函数。主要特性：
 * 
 * - 用于跨设备张量并行的分布式权重分区
 * - 自动张量形状处理和不同存储格式的转置
 * - 使用预计算三角值生成 RoPE（旋转位置嵌入）表
 * - 支持转置和非转置线性层权重格式
 * 
 * 权重分布策略：
 * - 全局张量（嵌入，归一化）：在所有设备上复制
 * - 注意力权重：按注意力头在设备间分区
 * - FFN 权重：按中间维度在设备间分区
 * - RoPE 表：通过数学计算按需生成
 * 
 * 所有函数返回 shared_ptr<Tensor> 以实现自动内存管理。
 */

#ifndef QWEN3_WEIGHT_HPP
#define QWEN3_WEIGHT_HPP

#include "qwen3_impl.hpp"

#include <cmath>
/*
 * 提取输入嵌入表
 * 
 * 为输入 token 嵌入查找表创建张量包装器。
 * 此表将 token ID 映射到其对应的稠密向量表示。
 * 
 * 张量属性：
 * - 形状：[dvoc, d] 其中 dvoc = 词汇表大小，d = 模型维度
 * - 数据类型：meta->dt_logits（通常为 FP16、BF16 或 FP32）
 * - 内存：引用全局权重存储（无复制）
 * - 分布：在所有设备上复制（未分区）
 * 
 * 用法：在推理过程中，token ID 用于索引此表
 * 以检索 transformer 处理前的初始隐藏表示。
 * 
 * 参数：
 * - meta：包含维度和数据类型的模型元数据
 * - w：包含所有模型参数的全局权重存储
 * 
 * 返回：输入嵌入表的共享张量包装器 [dvoc, d]
 */
inline std::shared_ptr<Tensor> getQwen3InEmbd(
    Qwen3Meta const *meta,
    Qwen3Weights const *w) {
    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape);
}


inline std::shared_ptr<Tensor> getQwen3AttnNorm(
    const Qwen3Meta *meta,
    const Qwen3Weights *w,
    size_t layer) {
    
    auto shape = std::vector<size_t>{meta->d};
    return Tensor::weight((char *)(w->attn_norm[layer]), w->dt_norm, shape);
}



/*
 * 提取最终层归一化权重
 * 
 * 为应用在语言模型头之前的最终 RMSNorm 层创建张量包装器。
 * 此归一化在词汇表投影之前稳定最终的隐藏表示。
 * 
 * 张量属性：
 * - 形状：[d] 其中 d = 模型隐藏维度
 * - 数据类型：w->dt_norm（归一化参数数据类型）
 * - 内存：引用全局权重存储
 * - 分布：在所有设备上复制
 * 
 * RMSNorm 公式：y = x / √(mean(x²) + ε) * γ
 * 其中 γ 是存储在此张量中的缩放参数。
 * 
 * 参数：
 * - meta：维度信息的模型元数据
 * - w：全局权重存储
 * 
 * 返回：输出归一化权重的共享张量包装器 [d]
 */

 
inline std::shared_ptr<Tensor> getQwen3OutNorm(
    Qwen3Meta const *meta,
    Qwen3Weights const *w) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)w->output_norm, w->dt_norm, shape);
}

/*
 * 提取输出嵌入 / 语言模型头权重
 * 
 * 为将隐藏状态映射到词汇表 logits 以进行下一个 token 预测的最终线性投影创建张量包装器。
 * 此层通常与输入嵌入表绑定以提高参数效率。
 * 
 * 权重格式处理：
 * - transpose_linear_weights = 0：权重存储为 [d, dvoc]（标准格式）
 * - transpose_linear_weights ≠ 0：权重存储为 [dvoc, d]（转置格式）
 * 
 * 当权重以转置方式存储时，我们应用 permute({1, 0}) 以获得
 * 正确的 [d, dvoc] 形状用于矩阵乘法：hidden_states @ weights。
 * 
 * 矩阵运算：logits = hidden_states @ weights
 * - hidden_states：[batch_size, d]
 * - weights：[d, dvoc] 
 * - logits：[batch_size, dvoc]
 * 
 * 张量属性：
 * - 最终形状：[d, dvoc] 无论存储格式如何
 * - 数据类型：meta->dt_logits
 * - 分布：在所有设备上复制
 * 
 * 参数：
 * - meta：维度的模型元数据
 * - w：带有格式标志的全局权重存储
 * 
 * 返回：输出投影权重的共享张量包装器 [d, dvoc]
 */
inline std::shared_ptr<Tensor> getQwen3OutEmbd(
    Qwen3Meta const *meta,
    Qwen3Weights const *w) {
    if (w->transpose_linear_weights != 0) {
        // 权重存储为 [dvoc, d]，需要转置为 [d, dvoc]
        auto shape = std::vector<size_t>({meta->dvoc, meta->d});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape)
            ->permute({1, 0});  // 转置：[dvoc, d] -> [d, dvoc]
    } else {
        // 权重已存储为 [d, dvoc]
        auto shape = std::vector<size_t>({meta->d, meta->dvoc});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape);
    }
}

/*
 * 提取特定层的注意力层归一化权重
 * 
 * 为每个 transformer 层中应用在注意力机制之前的 RMSNorm 权重创建张量包装器。
 * 这种注意力前归一化对训练稳定性和性能至关重要。
 * 
 * 张量属性：
 * - 形状：[d] 其中 d = 模型隐藏维度
 * - 数据类型：w->dt_norm（归一化数据类型）
 * - 分布：在所有设备上复制（所有设备相同）
 * - 用法：在注意力计算中的 QKV 投影之前应用
 * 
 * RMSNorm 应用：normalized = (x / √(mean(x²) + ε)) * scale_weights
 * 
 * 参数：
 * - meta：维度信息的模型元数据
 * - w：全局权重存储
 * - layer：层索引（0 到 nlayer-1）
 * 
 * 返回：注意力归一化权重的共享张量包装器 [d]
 */

inline std::shared_ptr<Tensor> getQwen3QNorm(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer) {
    // Q norm is applied per head dimension
    auto shape = std::vector<size_t>({meta->dh});
    return Tensor::weight((char *)(w->attn_q_norm[layer]), w->dt_norm, shape);
}

// Qwen3-specific: K normalization weights  
inline std::shared_ptr<Tensor> getQwen3KNorm(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer) {
    // K norm is applied per head dimension
    auto shape = std::vector<size_t>({meta->dh});
    return Tensor::weight((char *)(w->attn_k_norm[layer]), w->dt_norm, shape);
}




/*
 * 为分布式注意力提取 QKV 投影权重
 * 
 * 为查询、键、值投影权重创建设备特定的张量包装器。
 * 在分布式推理中，注意力头在设备间分区，
 * 因此每个设备获得总 QKV 投影矩阵的一个切片。
 */

/*
 * 为分布式推理提取 Q 投影权重
 * Q 投影：[2048, 2048] -> 每设备 [2048, 2048/ndev]
 */
inline std::shared_ptr<Tensor> getQwen3AttnQ(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    auto nh = meta->nh;         // 16 个查询头
    auto dh = meta->dh;         // 128 头维度
    auto d = meta->d;           // 2048 模型维度
    /*
     * Q 投影分布式切片计算：
     * - 总输出维度：nh * dh = 16 * 128 = 2048
     * - 每设备输出维度：(nh / ndev) * dh = (16/ndev) * 128
     * - 内存偏移：跳过前面设备的头数据
     */
    size_t heads_per_device = nh / ndev;                           // 每设备头数
    size_t output_dim_per_device = heads_per_device * dh;          // 每设备输出维度
    size_t offset = idev * output_dim_per_device * d * dsize(w->dt_mat);  // 字节偏移
    
    if (w->transpose_linear_weights != 0) {
        // 存储格式：[output_dim_per_device, d] -> 转置为 [d, output_dim_per_device]
        auto shape = std::vector<size_t>({output_dim_per_device, d});
        return Tensor::weight((char *)(w->attn_q_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        // 标准格式：[d, output_dim_per_device]
        auto shape = std::vector<size_t>({d, output_dim_per_device});
        return Tensor::weight((char *)(w->attn_q_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 为分布式推理提取 K 投影权重
 * 
 * K 投影：[1024, 2048] -> 每设备 [1024/ndev, 2048]
 */
inline std::shared_ptr<Tensor> getQwen3AttnK(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    auto nkvh = meta->nkvh;     // 8 个键值头
    auto dh = meta->dh;         // 128 头维度
    auto d = meta->d;           // 2048 模型维度
    
    /*
     * K 投影分布式切片计算：
     * - 总输出维度：nkvh * dh = 8 * 128 = 1024
     * - 每设备输出维度：(nkvh / ndev) * dh = (8/ndev) * 128
     */
    size_t kv_heads_per_device = nkvh / ndev;
    size_t output_dim_per_device = kv_heads_per_device * dh;
    size_t offset = idev * output_dim_per_device * d * dsize(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({output_dim_per_device, d});
        return Tensor::weight((char *)(w->attn_k_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, output_dim_per_device});
        return Tensor::weight((char *)(w->attn_k_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 为分布式推理提取 V 投影权重
 * 
 * V 投影：[1024, 2048] -> 每设备 [1024/ndev, 2048]
 * 与 K 投影完全相同的分布策略
 */
inline std::shared_ptr<Tensor> getQwen3AttnV(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    auto nkvh = meta->nkvh;     // 8 个键值头
    auto dh = meta->dh;         // 128 头维度  
    auto d = meta->d;           // 2048 模型维度
    
    /*
     * V 投影分布式切片计算（与 K 相同）
     */
    size_t kv_heads_per_device = nkvh / ndev;
    size_t output_dim_per_device = kv_heads_per_device * dh;
    size_t offset = idev * output_dim_per_device * d * dsize(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({output_dim_per_device, d});
        return Tensor::weight((char *)(w->attn_v_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, output_dim_per_device});
        return Tensor::weight((char *)(w->attn_v_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 为分布式推理提取注意力输出投影权重
 * 
 * 为注意力输出投影创建设备特定的张量包装器，
 * 该投影将多头注意力输出合并回模型维度。
 * 在分布式推理中，此投影采用来自此设备头的连接注意力输出，
 * 并将其投影到完整的模型维度。
 * 
 * 权重分区策略：
 * - 每个设备处理 nh/ndev 个注意力头
 * - 输入维度：nh/ndev * dh（此设备上的注意力头）
 * - 输出维度：d（完整模型维度，所有设备相同）
 * - 结果通过 all-reduce 操作在设备间求和
 * 
 * 内存布局和偏移计算：
 * - 全局权重形状：[nh * dh, d] 或转置
 * - 每设备切片：[nh/ndev * dh, d]  
 * - 字节偏移：idev * (nh/ndev * dh) * d * sizeof(data_type)
 * 
 * 矩阵运算：output = attention_heads @ output_weights
 * - attention_heads：[batch_size, nh/ndev * dh]（设备特定）
 * - output_weights：[nh/ndev * dh, d]（设备特定切片）
 * - output：[batch_size, d]（完整模型维度）
 * 
 * 分布式计算：
 * 1. 每个设备计算其注意力头的部分输出
 * 2. All-reduce 在设备间求和部分输出
 * 3. 最终结果表示完整的注意力输出
 * 
 * 参数：
 * - meta：维度的模型元数据
 * - w：全局权重存储
 * - layer：Transformer 层索引
 * - idev：当前设备索引
 * - ndev：设备总数
 * 
 * 返回：设备特定的注意力输出权重 [nh/ndev * dh, d] 或 [d, nh/ndev * dh]
 */
inline std::shared_ptr<Tensor> getQwen3AttnO(Qwen3Meta const *meta,
                                        Qwen3Weights const *w, size_t layer,
                                        size_t idev, size_t ndev) {
    // 提取模型维度
    auto nh = meta->nh;         // 总查询头数
    auto dh = meta->dh;         // 头维度  
    auto d = meta->d;           // 模型隐藏维度
    
    /*
     * 计算设备切片的内存偏移
     * 
     * 每个设备获得对应其分配的注意力头的输出投影切片：
     * offset = device_index * heads_per_device * head_dim * model_dim * data_size
     */
    size_t offset = idev * d * (nh / ndev * dh) * dsize(w->dt_mat);
    
    /*
     * 处理不同的权重存储格式
     */
    if (w->transpose_linear_weights != 0) {
        // 权重存储为 [d, nh/ndev * dh]，转置为 [nh/ndev * dh, d]
        auto shape = std::vector<size_t>({d, nh / ndev * dh});
        return Tensor::weight((char *)(w->attn_o_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});  // 转置到正确方向
    } else {
        // 权重已处于正确格式：[nh/ndev * dh, d]
        auto shape = std::vector<size_t>({nh / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_o_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 提取特定层的 FFN 层归一化权重
 * 
 * 为每个 transformer 层中应用在前馈网络之前的 RMSNorm 权重创建张量包装器。
 * 这种 FFN 前归一化稳定训练并提高性能。
 * 
 * 张量属性：
 * - 形状：[d] 其中 d = 模型隐藏维度
 * - 数据类型：w->dt_norm（归一化参数数据类型）
 * - 分布：在所有设备上复制（所有设备相同）
 * - 用法：在 FFN 门控/上升投影之前应用
 * 
 * RMSNorm 公式：y = (x / √(mean(x²) + ε)) * scale_weights
 * 这在 FFN 处理之前归一化注意力后的隐藏状态。
 * 
 * 参数：
 * - meta：维度信息的模型元数据
 * - w：全局权重存储
 * - layer：层索引（0 到 nlayer-1）
 * 
 * 返回：FFN 归一化权重的共享张量包装器 [d]
 */
inline std::shared_ptr<Tensor> getQwen3MLPNorm(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->mlp_norm[layer]), w->dt_norm, shape);
}

/*
 * 为分布式推理提取 FFN 门控和上升投影权重
 * 
 * 为用于 SwiGLU 激活函数的组合门控和上升投影创建设备特定的张量包装器。
 * 这些投影将模型维度扩展到中间 FFN 维度。
 * 
 * SwiGLU 架构：
 * - 门控投影：用于门控机制的线性变换
 * - 上升投影：用于值计算的线性变换  
 * - 组合操作：gate_output = gate_proj(x), up_output = up_proj(x)
 * - SwiGLU 激活：output = gate_output * swish(up_output)
 * 
 * 权重分区策略：
 * - 总中间维度：di（FFN 扩展因子 * model_dim）
 * - 每设备维度：di/ndev（在设备间分布）
 * - 门控 + 上升组合：2 * di/ndev（两个投影连接）
 * - 每个设备处理中间维度的一个切片
 * 
 * 内存布局：
 * - 全局权重形状：[d, 2*di] 或转置
 * - 每设备切片：[d, 2*di/ndev]
 * - 连接格式：沿输出维度的 [gate_weights, up_weights]
 * - 字节偏移：idev * (2*di/ndev) * d * sizeof(data_type)
 * 
 * 矩阵运算：
 * [gate_output, up_output] = hidden_states @ [gate_weights, up_weights]
 * - hidden_states：[batch_size, d]
 * - combined_weights：[d, 2*di/ndev] 
 * - outputs：[batch_size, 2*di/ndev] = [gate_batch, up_batch]
 * 
 * 参数：
 * - meta：维度的模型元数据
 * - w：全局权重存储
 * - layer：Transformer 层索引
 * - idev：当前设备索引
 * - ndev：设备总数
 * 
 * 返回：设备特定的 FFN 门控和上升权重 [d, 2*di/ndev] 或 [2*di/ndev, d]
 */
/*
 * 为分布式推理提取 FFN Gate 投影权重
 * 
 * Gate 投影：[6144, 2048] -> 每设备 [6144/ndev, 2048]
 */
inline std::shared_ptr<Tensor> getQwen3MLPGate(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    auto di = meta->di;         // 6144 中间维度
    auto d = meta->d;           // 2048 模型维度
    
    /*
     * Gate 投影分布式切片计算：
     * - 总输出维度：di = 6144
     * - 每设备输出维度：di / ndev = 6144/ndev
     * - 内存偏移：跳过前面设备的中间维度数据
     */

    size_t intermediate_per_device = di / ndev;
    size_t offset = idev * intermediate_per_device * d * dsize(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        // 存储格式：[intermediate_per_device, d] -> 转置为 [d, intermediate_per_device]
        auto shape = std::vector<size_t>({intermediate_per_device, d});
        return Tensor::weight((char *)(w->mlp_gate_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        // 标准格式：[d, intermediate_per_device]
        auto shape = std::vector<size_t>({d, intermediate_per_device});
        return Tensor::weight((char *)(w->mlp_gate_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 为分布式推理提取 FFN Up 投影权重
 * 
 * Up 投影：[6144, 2048] -> 每设备 [6144/ndev, 2048]
 * 与 Gate 投影完全相同的分布策略
 */
inline std::shared_ptr<Tensor> getQwen3MLPUp(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    auto di = meta->di;         // 6144 中间维度
    auto d = meta->d;           // 2048 模型维度
    
    /*
     * Up 投影分布式切片计算（与 Gate 相同）
     * - 总输出维度：di = 6144
     * - 每设备输出维度：di / ndev = 6144/ndev
     */
    size_t intermediate_per_device = di / ndev;
    size_t offset = idev * intermediate_per_device * d * dsize(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({intermediate_per_device, d});
        return Tensor::weight((char *)(w->mlp_up_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, intermediate_per_device});
        return Tensor::weight((char *)(w->mlp_up_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 为分布式推理提取 FFN 下降投影权重
 * 
 * 为将中间 FFN 维度映射回模型维度的下降投影创建设备特定的张量包装器。
 * 这在 SwiGLU 激活后完成 FFN 计算。
 * 
 * FFN 下降投影作用：
 * - 输入：SwiGLU 激活的中间表示 [batch, di/ndev]
 * - 输出：回到模型维度的隐藏状态 [batch, d]  
 * - 目的：将扩展的中间特征投影回残差流
 * 
 * 权重分区策略：
 * - 输入维度：di/ndev（每设备中间维度）
 * - 输出维度：d（完整模型维度，所有设备相同）
 * - 每个设备处理其中间维度的切片
 * - 结果通过 all-reduce 在设备间求和
 * 
 * 内存布局：
 * - 全局权重形状：[di, d] 或转置
 * - 每设备切片：[di/ndev, d]
 * - 字节偏移：idev * (di/ndev) * d * sizeof(data_type)
 * 
 * 矩阵运算：output = intermediate_activated @ down_weights
 * - intermediate_activated：[batch_size, di/ndev]（SwiGLU 后）
 * - down_weights：[di/ndev, d]（设备特定切片）
 * - output：[batch_size, d]（回到模型维度）
 * 
 * 分布式计算：
 * 1. 每个设备计算其中间切片的部分输出
 * 2. All-reduce 在设备间求和部分输出  
 * 3. 最终结果表示完整的 FFN 输出
 * 
 * 参数：
 * - meta：维度的模型元数据
 * - w：全局权重存储
 * - layer：Transformer 层索引
 * - idev：当前设备索引
 * - ndev：设备总数
 * 
 * 返回：设备特定的 FFN 下降权重 [di/ndev, d] 或 [d, di/ndev]
 */



/*
 * 为分布式推理提取 FFN Down 投影权重
 * 
 * Down 投影：[2048, 6144] -> 每设备 [2048, 6144/ndev]
 * 注意：Down 的分布策略与 Gate/Up 不同，它是按输入维度切片
 */
inline std::shared_ptr<Tensor> getQwen3MLPDown(
    Qwen3Meta const *meta,
    Qwen3Weights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    auto di = meta->di;         // 6144 中间维度
    auto d = meta->d;           // 2048 模型维度
    
    /*
     * Down 投影分布式切片计算：
     * - 总输入维度：di = 6144（从 Gate/Up 的输出接收）
     * - 每设备输入维度：di / ndev = 6144/ndev
     * - 输出维度：d = 2048（完整模型维度）
     * 注意：Down 投影的权重矩阵是 [d, di]，但我们需要按输入维度切片
     * 所以实际的权重切片是 [d, di/ndev]
     */
    size_t intermediate_per_device = di / ndev;
    size_t offset = idev * d * intermediate_per_device * dsize(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        // 存储格式：[d, intermediate_per_device] -> 转置为 [intermediate_per_device, d]
        auto shape = std::vector<size_t>({d, intermediate_per_device});
        return Tensor::weight((char *)(w->mlp_down_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        // 标准格式：[intermediate_per_device, d]
        auto shape = std::vector<size_t>({intermediate_per_device, d});
        return Tensor::weight((char *)(w->mlp_down_proj[layer]) + offset, w->dt_mat, shape);
    }
}


/*
 * 生成 RoPE 正弦查找表
 * 
 * 为旋转位置嵌入（RoPE）创建预计算的正弦表。
 * RoPE 通过将旋转矩阵应用于查询和键向量来编码位置信息，
 * 基于它们在序列中的位置。
 * 
 * RoPE 数学基础：
 * 对于位置 pos 和维度对 (i, i+d/2)：
 * - 旋转角度：θ_i = pos / (base^(2i/d)) 其中 base = meta->theta（通常为 10000）
 * - 正弦分量：sin(θ_i) = sin(pos / (base^(2i/d)))
 * 
 * 表结构：
 * - 形状：[dctx, dh/2] 其中 dctx = 最大上下文长度，dh = 头维度
 * - 条目 [pos][i]：sin(pos / (theta^(2i/dh))) 对于位置 pos 和维度 i
 * - 覆盖所有可能的位置（0 到 dctx-1）和头维度的一半
 * 
 * 数据类型处理：
 * - 支持 FP16、BF16 和 FP32 基于 meta->dt_logits
 * - 从 float32 计算转换为目标数据类型
 * - 使用适当的转换函数（f32_to_f16、f32_to_bf16）
 * 
 * 内存管理：
 * - 为计算分配临时存储
 * - 创建带有计算值副本的张量包装器
 * - 在张量创建后释放临时存储
 * 
 * 推理期间的用法：
 * - 由位置 ID 查找以获取旋转的正弦值
 * - 与余弦表结合形成完整的旋转矩阵
 * - 应用于查询和键向量
 * 
 * 参数：
 * - meta：包含上下文长度、头维度、theta 和数据类型的模型元数据
 * 
 * 返回：包含预计算正弦值的共享张量 [dctx, dh/2]
 */
inline std::shared_ptr<Tensor> getQwen3SinTable(Qwen3Meta const *meta) {
    auto half_dh = meta->dh / 2;                    // 头维度的一半
    auto unit = dsize(meta->dt_logits);             // 数据类型的大小（字节）
    void *table = std::malloc(meta->dctx * half_dh * unit);  // 分配临时存储

    /*
     * 为所有位置和维度计算正弦值
     * 
     * 嵌套循环结构：
     * - 外循环：遍历所有可能的位置（0 到 dctx-1）
     * - 内循环：遍历头维度的一半（0 到 dh/2-1）
     */
    for (size_t i = 0; i < meta->dctx; i++) {      // 位置循环
        for (size_t j = 0; j < half_dh; j++) {     // 维度循环
            /*
             * RoPE 正弦计算：
             * 
             * 公式：sin(position / (theta^(2*dim_index / head_dimension)))
             * - position：i（当前序列位置）
             * - dim_index：j（当前维度索引）  
             * - theta：meta->theta（旋转基数，通常为 10000）
             * - head_dimension：meta->dh（完整头维度）
             */
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            
            /*
             * 根据目标数据类型转换和存储
             * 
             * 以 FP32 计算以保证精度，然后转换为目标格式。
             */
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _sin;
            } else {
                std::cout << "不支持的数据类型" << std::endl;
                exit(1);
            }
        }
    }
    
    // 创建张量包装器并清理临时存储
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);  // 释放临时存储（张量有自己的副本）
    return tensor;
}

/*
 * 生成 RoPE 余弦查找表
 * 
 * 为旋转位置嵌入（RoPE）创建预计算的余弦表。
 * 这补充了正弦表，形成完整的旋转矩阵以编码
 * transformer 注意力机制中的位置信息。
 * 
 * RoPE 数学基础：
 * 对于位置 pos 和维度对 (i, i+d/2)：
 * - 旋转角度：θ_i = pos / (base^(2i/d)) 其中 base = meta->theta
 * - 余弦分量：cos(θ_i) = cos(pos / (base^(2i/d)))
 * 
 * 表结构：
 * - 形状：[dctx, dh/2] 其中 dctx = 最大上下文长度，dh = 头维度
 * - 条目 [pos][i]：cos(pos / (theta^(2i/dh))) 对于位置 pos 和维度 i
 * - 与正弦表相同的结构以实现高效配对查找
 * 
 * RoPE 旋转矩阵应用：
 * 对于每个位置和维度对 (i, i+dh/2)：
 * - q'[i] = q[i] * cos(θ) - q[i+dh/2] * sin(θ)
 * - q'[i+dh/2] = q[i] * sin(θ) + q[i+dh/2] * cos(θ)
 * 
 * 这在 (i, i+dh/2) 平面中创建一个 2D 旋转，直接编码
 * 注意力计算中的相对位置信息。
 * 
 * 数据类型支持：
 * - FP16：半精度以提高内存效率
 * - BF16：脑浮点以获得更好的数值范围  
 * - FP32：全精度以获得最大准确性
 * - 以 FP32 计算然后转换为目标格式
 * 
 * 内存管理：
 * - 计算期间的临时分配
 * - 持久存储的张量副本
 * - 临时内存的自动清理
 * 
 * 性能考虑：
 * - 预计算表避免推理期间昂贵的三角运算
 * - 注意力计算期间缓存友好的访问模式
 * - 在所有层和注意力头间共享
 * 
 * 参数：
 * - meta：带有上下文长度、维度、旋转基数和数据类型的模型元数据
 * 
 * 返回：包含预计算余弦值的共享张量 [dctx, dh/2]
 */
inline std::shared_ptr<Tensor> getQwen3CosTable(Qwen3Meta const *meta) {
    auto half_dh = meta->dh / 2;                    // 头维度的一半
    auto unit = dsize(meta->dt_logits);             // 数据类型的大小（字节）
    void *table = std::malloc(meta->dctx * half_dh * unit);  // 分配临时存储

    /*
     * 为所有位置和维度计算余弦值
     * 
     * 与正弦表相同的循环结构以保持一致性和配对访问。
     */
    for (size_t i = 0; i < meta->dctx; i++) {      // 位置循环
        for (size_t j = 0; j < half_dh; j++) {     // 维度循环  
            /*
             * RoPE 余弦计算：
             * 
             * 公式：cos(position / (theta^(2*dim_index / head_dimension)))
             * - 与正弦计算相同但使用余弦函数
             * - 确保 sin² + cos² = 1 以获得正确的旋转矩阵
             */
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            
            /*
             * 根据目标数据类型转换和存储
             * 
             * 与正弦表相同的数据类型处理以保持一致。
             */
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _cos;
            } else {
                std::cout << "不支持的数据类型" << std::endl;
                exit(1);
            }
        }
    }
    
    // 创建张量包装器并清理临时存储
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);  // 释放临时存储（张量有自己的副本）
    return tensor;
}

#endif