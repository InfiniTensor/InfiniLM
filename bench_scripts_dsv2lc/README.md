# DeepSeek-V2-Lite-Chat 性能测试说明

这个目录用于对比 DeepSeek-V2-Lite-Chat 在 PyTorch Transformers 和 InfiniLM 上的推理性能。

## 完整测试流程

推荐流程是：运行一次完整测试矩阵；脚本会让每个 backend/case 在同一个日志文件里重复测试 3 次，然后对同一个日志里的 3 次推理时间取平均。

```bash
ssh 28_hygon_bw1000
docker exec -it baoming_test bash

cd /home/libaoming/workplace/InfiniLM_latest/bench_scripts_dsv2lc
source /home/libaoming/venvs/deepseek-v2/bin/activate

# 默认 REPEAT=3。bs16_4096_4096 使用双卡，其余 case 使用单卡。
SINGLE_GPU=0 \
TP2_GPUS=0,1 \
MAX_WAIT_TIME=3600s \
./run_serial_dsv2lc.sh
```

如果只是快速验证，可以把重复次数改成 1：

```bash
REPEAT=1 SINGLE_GPU=0 TP2_GPUS=0,1 MAX_WAIT_TIME=3600s ./run_serial_dsv2lc.sh
```

跑完后会生成一个时间戳目录：

```text
serial_logs_YYYYMMDD_HHMMSS
```

每个 case 只有一份 Torch 日志和一份 InfiniLM 日志，例如：

```text
torch_bench_dsv2lc_16_4096_4096.log
infinilm_bench_dsv2lc_16_4096_4096.log
```

同一个日志里会包含多次重复测试：

```text
[repeat_start] ... run=1/3
... total_time: ... ms
[repeat_finish] ... run=1/3 status=0
[repeat_start] ... run=2/3
... total_time: ... ms
[repeat_finish] ... run=2/3 status=0
[repeat_start] ... run=3/3
... total_time: ... ms
[repeat_finish] ... run=3/3 status=0
```

运行结果对比：

```bash
cd /home/libaoming/workplace/InfiniLM_latest/bench_scripts_dsv2lc

LOGDIR=serial_logs_YYYYMMDD_HHMMSS

for case in 4_1024_1024 4_4096_4096 16_128_128 16_1024_1024 16_4096_4096; do
  python calculate_performance.py --inputs \
    "$LOGDIR/torch_bench_dsv2lc_${case}.log" \
    "$LOGDIR/infinilm_bench_dsv2lc_${case}.log"
done | tee "$LOGDIR/compare_all.txt"
```

平均方式：

- `calculate_performance.py` 会从同一个日志里提取所有 `total_time`。
- 先分别计算 Torch / InfiniLM 的平均 `total_time`。
- 再计算 `time_speedup = torch_avg_time / infinilm_avg_time`。
- 也就是说，先平均推理时间，再算性能加速比。
- 不建议先算每次重复的 speedup 再取平均，因为 speedup 是比值，直接平均容易被单次抖动放大。

## 环境准备

测试需要在 28 机器的 `baoming_test` 容器里执行：

```bash
ssh 28_hygon_bw1000
docker exec -it baoming_test bash

cd /home/libaoming/workplace/InfiniLM_latest/bench_scripts_dsv2lc
source /home/libaoming/venvs/deepseek-v2/bin/activate
```

长 case 开跑前建议先确认显卡空闲：

```bash
/opt/dtk/bin/rocm-smi --showuse --showmemuse
ps -ef | grep -E 'bench.py|pytorch_bench.py|python' | grep -v grep
```

## 串行测试脚本

当前推荐使用 `run_serial_dsv2lc.sh`。这个脚本串行执行 case，避免多个测试并发争抢 CPU/GPU 资源。

默认 case：

| case | 设备 |
|---|---|
| bs4_1024_1024 | 单卡 |
| bs4_4096_4096 | 单卡 |
| bs16_128_128 | 单卡 |
| bs16_1024_1024 | 单卡 |
| bs16_4096_4096 | 双卡 TP2 |

跑一轮完整测试矩阵，每个测试重复 3 次：

```bash
cd /home/libaoming/workplace/InfiniLM_latest/bench_scripts_dsv2lc

SINGLE_GPU=0 TP2_GPUS=0,1 MAX_WAIT_TIME=3600s ./run_serial_dsv2lc.sh
```

常用参数：

```bash
# 重复次数，默认是 3。
REPEAT=1 SINGLE_GPU=0 TP2_GPUS=0,1 ./run_serial_dsv2lc.sh

# 修改单次测试超时时间，默认是 1800s。
MAX_WAIT_TIME=3600s SINGLE_GPU=0 TP2_GPUS=0,1 ./run_serial_dsv2lc.sh

# 只跑 PyTorch Transformers。
RUN_INFINILM=0 SINGLE_GPU=0 TP2_GPUS=0,1 ./run_serial_dsv2lc.sh

# 只跑 InfiniLM。
RUN_TORCH=0 SINGLE_GPU=0 TP2_GPUS=0,1 ./run_serial_dsv2lc.sh

# 指定日志目录。
OUT_DIR=serial_logs_manual SINGLE_GPU=0 TP2_GPUS=0,1 ./run_serial_dsv2lc.sh
```

脚本里的重要默认参数：

```bash
MODEL=/home_aclsylqidf/shared/DeepSeek-V2-Lite-Chat
REPEAT=3
DSV2_TORCH_ATTN=flash_attention_2
DSV2_TORCH_DEVICE_MAP=dsv2_lmhead0
DSV2_TORCH_SPLIT_LAYER=13
DSV2_TORCH_MAX_MEMORY=0:60GiB,1:60GiB
DSV2_INFINI_DEVICE=hygon
DSV2_INFINI_WEIGHT_LOAD=sync
```

说明：

- `bs16_4096_4096` 默认使用 `TP2_GPUS=0,1` 和 `--tp 2`。
- InfiniLM 只有双卡 TP2 case 会打开 `--enable-paged-attn`。
- 其他 case 默认使用 `SINGLE_GPU=0` 单卡运行。
- `MAX_WAIT_TIME` 是单次重复测试的超时时间，不是整个日志文件的总超时时间。

## 日志怎么看

每个日志开头会记录 backend、case、devices、TP 配置和环境变量：

```text
[start] 2026-07-08 09:41:34 backend=torch case=dsv2lc_4_1024_1024 devices=0 tp=1
[env] CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
```

PyTorch 日志重点看：

```text
total_time: ... ms
decode/output throughput: ... tok/s
```

InfiniLM 日志重点看：

```text
Generation completed in ... ms
Prefill TTFT: ... ms  Throughput: ... tok/s
Decode  Avg ITL: ... ms   Throughput: ... tok/s
total_time: ... ms
```

如果 case 失败，优先检查：

```text
Traceback
OutOfMemory
cudaMalloc
[repeat_finish] ... status=非0
[finish] ... status=非0
```

## 旧脚本说明

`run_all_dsv2lc.sh` 和 `compare_all_dsv2lc.sh` 是早期固定 case 脚本。现在推荐使用 `run_serial_dsv2lc.sh`，再用 `calculate_performance.py` 汇总同一个日志里的多次重复测试结果。

## 单个独立脚本执行命令

如果只想跑某一个固定 case，可以直接执行对应的 `.py` 脚本。每组命令都可以直接复制粘贴。

### bs4_2k_3k

```bash
cd /home/libaoming/workplace/InfiniLM_latest/bench_scripts_dsv2lc

timeout 3600s python infinilm_bench_dsv2lc_4_2k_3k.py > infinilm_bench_dsv2lc_4_2k_3k.log 2>&1
timeout 3600s python pytorch_bench_dsv2lc_4_2k_3k.py > torch_bench_dsv2lc_4_2k_3k.log 2>&1

python calculate_performance.py --inputs \
  torch_bench_dsv2lc_4_2k_3k.log \
  infinilm_bench_dsv2lc_4_2k_3k.log
```

### bs4_8k_10k

```bash
cd /home/libaoming/workplace/InfiniLM_latest/bench_scripts_dsv2lc

timeout 3600s python infinilm_bench_dsv2lc_4_8k_10k.py > infinilm_bench_dsv2lc_4_8k_10k.log 2>&1
timeout 3600s python pytorch_bench_dsv2lc_4_8k_10k.py > torch_bench_dsv2lc_4_8k_10k.log 2>&1

python calculate_performance.py --inputs \
  torch_bench_dsv2lc_4_8k_10k.log \
  infinilm_bench_dsv2lc_4_8k_10k.log
```

### bs16_200_400

```bash
cd /home/libaoming/workplace/InfiniLM_latest/bench_scripts_dsv2lc

timeout 3600s python infinilm_bench_dsv2lc_16_200_400.py > infinilm_bench_dsv2lc_16_200_400.log 2>&1
timeout 3600s python pytorch_bench_dsv2lc_16_200_400.py > torch_bench_dsv2lc_16_200_400.log 2>&1

python calculate_performance.py --inputs \
  torch_bench_dsv2lc_16_200_400.log \
  infinilm_bench_dsv2lc_16_200_400.log
```

### bs16_2k_3k

```bash
cd /home/libaoming/workplace/InfiniLM_latest/bench_scripts_dsv2lc

timeout 3600s python infinilm_bench_dsv2lc_16_2k_3k.py > infinilm_bench_dsv2lc_16_2k_3k.log 2>&1
timeout 3600s python pytorch_bench_dsv2lc_16_2k_3k.py > torch_bench_dsv2lc_16_2k_3k.log 2>&1

python calculate_performance.py --inputs \
  torch_bench_dsv2lc_16_2k_3k.log \
  infinilm_bench_dsv2lc_16_2k_3k.log
```

### bs16_8k_10k

```bash
cd /home/libaoming/workplace/InfiniLM_latest/bench_scripts_dsv2lc

timeout 3600s python infinilm_bench_dsv2lc_16_8k_10k.py > infinilm_bench_dsv2lc_16_8k_10k.log 2>&1
timeout 3600s python pytorch_bench_dsv2lc_16_8k_10k.py > torch_bench_dsv2lc_16_8k_10k.log 2>&1

python calculate_performance.py --inputs \
  torch_bench_dsv2lc_16_8k_10k.log \
  infinilm_bench_dsv2lc_16_8k_10k.log
```

