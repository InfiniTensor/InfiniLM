#ifndef __CDIST_CPU_H__
#define __CDIST_CPU_H__

#include "../cdist.h"

// 使用 cdist.h 中定义的 DESCRIPTOR 宏
// 这将在命名空间 op::cdist::cpu 中生成 Descriptor 类
// 该类包含对 CdistInfo 的引用以及 create/calculate 等接口
DESCRIPTOR(cpu)

#endif // __CDIST_CPU_H__
