#!/bin/bash
#SBATCH --job-name=test_job              # 任务名
#SBATCH --output=output_%j.log           # 标准输出文件（%j 会替换成 job ID）
#SBATCH --error=error_%j.log             # 标准错误输出文件
#SBATCH --partition=nvidia               # 分区名（机器系统默认分区是 nvidia）
#SBATCH --nodes=1                        # 需要的节点数
#SBATCH --ntasks=1                       # 总任务数（通常 = 节点数 × 每节点任务数）
#SBATCH --cpus-per-task=8               # 每个任务需要的 CPU 核心数
#SBATCH --gres=gpu:nvidia:4             # 请求 4 块 GPU（nvidia 是 Gres 类型）
#SBATCH --mem=32G                       # 请求的内存

# 需要用到计算资源的命令
# 推荐使用 srun 启动主程序，自动绑定资源
source /data/apps/env.sh
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate py313
export INFINI_ROOT=$HOME/.infini 
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH
export PATH="/data/apps/xmake/bin:/usr/local/cuda/bin:$PATH"

export PYTHONPATH=$HOME/InfiniLM/scripts:$PYTHONPATH

cd $HOME/InfiniLM

#srun python scripts/qwen3vl_test.py
#srun python scripts/qwen3vl.py --nvidia /data/shared/models/Qwen3-VL-2B-Instruct
srun python scripts/launch_server.py --model-path /data/shared/models/Qwen3-VL-2B-Instruct --dev nvidia --ndev 4