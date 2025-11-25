#ifndef DEEPSEEK_V3_WEIGHTS_H
#define DEEPSEEK_V3_WEIGHTS_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stddef.h>
#include <stdint.h>

struct LLaDAMoEModel; 

typedef struct {

} LLaDAMoEMeta;


typedef struct{
} JiugeWeights;

// 改进后的权重加载器结构体
typedef struct {

} LLaDAMoEWeightLoader; // # TODO: 



__C void
createLLaDAMoEModel();

__C void
destroyLaDAMoEModel();

__C void
inferBatchLLaDAMoE();

__C void
forwardBatchLLaDAMoE();
