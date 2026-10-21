# Kimi-K3 草稿成本实验，非在线推理成绩

`dspark_markov_r4163.hip` 是独立 pure HIP Markov 头候选。仅支持 BF16、vocab 163840、rank 256、B1/2/4/8、gfx936；输入必须在实际 owner GPU，且为约定连续布局。使用 FP32 累积，先按 BF16 舍入 Markov bias，再与 BF16 base logits 相加并舍入。不能据此替换其他模型的算子或目标模型数学路径。

在 DTK 环境可独立编译：

```sh
hipcc -std=c++17 -O3 -fPIC -shared --offload-arch=gfx936 -fno-fast-math \
  dspark_markov_r4163.hip -o libdspark_markov_r4163.so
```

`run_dspark_profile_r4154.py` / `audit_dspark_profile_r4156.py` 保存原 Torch 分阶段与图重放测试；`run_dspark_fused_r4164.py` / `audit_dspark_fused_r4165.py` 保存融合候选测试和独立审计。`dspark_fused_r4164_inside.sh` 是原容器环境包装器，路径是部署占位约定，不含访问凭据。迁移时按整个 handoff 重建目标依赖、准备已审核草稿源码包和对应 SHA，再映射权重、历史 fixture 与输出目录；这些驱动不是可直接运行的通用服务。

不附带权重或原始大文件。草稿 revision、SHA、计时与范围见 [精简证据](../../evidence/speculative-decode-components-r4183.json)。这只是固定前缀 draft 组件，最新 target feature 接口、在线 target verify 和事务提交/回滚仍需实现。K16 属于未训练长度外推；不能把 proposal tok/s 或这张耗时表当成单用户最终 tok/s。
