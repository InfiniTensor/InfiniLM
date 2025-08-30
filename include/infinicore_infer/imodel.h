// imodel.h
#pragma once

// 只需要 KVCache 的前向声明，不需要知道它的具体实现
struct KVCache; 

// 这是所有模型都必须遵守的通用接口
class IModel {
public:
    // C++ 接口类必须有虚析构函数
    virtual ~IModel() = default;

    // 定义所有模型都必须提供的功能作为“纯虚函数” (= 0)
    // 任何继承 IModel 的类都必须自己实现这些函数
    
    // 创建一个适用于此模型的 KVCache 结构
    virtual KVCache* createKVCache() const = 0;

    // 复制 KVCache（例如用于 beam search）
    virtual KVCache* duplicateKVCache(const KVCache* cache, unsigned int seq_len) const = 0;
    
    // 销毁 KVCache
    virtual void dropKVCache(KVCache* cache) const = 0;
};