#ifndef __SMOOTH_L1_LOSS_MOORE_API_H__
#define __SMOOTH_L1_LOSS_MOORE_API_H__

// 引入上层定义的 Descriptor 宏和基础类
#include "../smooth_l1_loss.h"

// 使用 smooth_l1_loss.h 中定义的 DESCRIPTOR 宏
// 这将自动生成 op::smooth_l1_loss::moore::Descriptor 类定义
DESCRIPTOR(moore)

#endif // __SMOOTH_L1_LOSS_MOORE_API_H__
