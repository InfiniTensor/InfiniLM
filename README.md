# InfiniCore

InfiniCore是一个跨平台统一编程工具集，为不同芯片平台的功能（包括计算、运行时、通信等）提供统一 C 语言接口。目前支持的芯片包括CPU、英伟达GPU、华为昇腾NPU、寒武纪MLU、摩尔线程GPU、天数智芯GPU、沐曦GPU、曙光DCU、昆仑芯。

## 一、使用说明

### 1. 配置

#### 查看当前配置

```xmake
xmake f -v
```

#### 配置 CPU （默认配置）

```xmake
xmake f -cv
```

#### 配置加速卡

```xmake
# 英伟达
# 可以指定 CUDA 路径， 一般环境变量为 `CUDA_HOME` 或者 `CUDA_ROOT`
xmake f --nv-gpu=true --cuda=$CUDA_HOME -cv

# 寒武纪
xmake f --cambricon-mlu=true -cv

# 华为昇腾
xmake f --ascend-npu=true -cv
```

### 2. 编译安装

```xmake
xmake build && xmake install
# 默认安装路径为 $HOME/.infini
```

### 3. 设置环境变量

按输出提示设置 `INFINI_ROOT` 和 `LD_LIBRARY_PATH` 环境变量。

### 4. 运行算子测试

```bash
python test/infiniop/[operator].py [--cpu | --nvidia | --cambricon | --ascend]
```
