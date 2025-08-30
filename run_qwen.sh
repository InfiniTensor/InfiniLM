#!/bin/bash

#================================================================
# Part 1: Slurm 配置指令 -- 告诉 Slurm 如何运行你的任务
#================================================================

#-- 设置任务的基本信息
#SBATCH --job-name=my_pytorch_job     # 任务名，请修改成你自己的，方便识别
#SBATCH --output=slurm_out_%j.log     # 指定标准输出文件，%j 会被替换为作业ID
#SBATCH --error=slurm_err_%j.log      # 指定错误输出文件

#-- 设置任务的资源需求 (这是你需要修改的核心部分)
#SBATCH --partition=mx                # 分区名，根据手册，固定写 mx
#SBATCH --nodes=1                     # 节点数，根据手册，固定写 1
#SBATCH --ntasks=1                    # 总任务数，根据手册，固定写 1
#SBATCH --gres=gpu:mx:8               # 【重要】需要的GPU数量，例如 :1, :2, :4
#SBATCH --cpus-per-task=16            # 【重要】需要的CPU核心数 (最大32)
#SBATCH --mem=128G                    # 【重要】需要的内存大小 (最大256G)
#SBATCH --time=00:20:00               # 【重要】任务运行时间上限 (HH:MM:SS)，默认10分钟，最大20分钟

#================================================================
# Part 2: 执行你的命令 -- 告诉计算节点具体要做什么
#================================================================
#-- 打印一些有用的信息到输出文件
echo "========================================================"
echo "Job ID: InfiniCore-Qwen3-1.7B"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $SLURM_GPUS"
echo "Job Started at: $(date)"
echo "========================================================"
echo ""

#-- 1. 激活你的环境 (如果使用 Conda 或 venv)
# source /

#-- 2. 切换到你的代码目录 (推荐使用绝对路径)
cd /home/hootandy/InfiniLM

#-- 3. 运行你的主程序
#    手册推荐使用 srun 来启动，这样可以更好地绑定资源
#    在下面替换成你自己的 python 脚本和参数
echo "Running python script..."
srun python scripts/qwen.py --metax /home/shared/models/Qwen3-1.7B/ 8

#-- 任务结束，打印信息
echo ""
echo "========================================================"
echo "Job Finished at: $(date)"
echo "========================================================"



