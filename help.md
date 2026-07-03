NV
InfiniCore
1. 基础
source /data/shared/spack/share/spack/setup-env.sh
spack load cmake@4.0.0
spack load llvm@16.0.6
spack load cuda@12.9.0
spack load cudnn@9.8.0.87-12
spack load nccl@2.27.5-1
spack find --loaded
source /home/mayuhang/env/.venv/bin/activate


xmake f --nv-gpu=y --ccl=y -cv
xmake build && xmake install
xmake build _infinicore
xmake install _infinicore
pip install -e .
2. fla
source /data/shared/spack/share/spack/setup-env.sh
spack load cmake@4.0.0
spack load llvm@16.0.6
spack load cuda@12.9.0
spack load cudnn@9.8.0.87-12
spack load nccl@2.27.5-1
spack find --loaded
source /home/mayuhang/env/.venv/bin/activate


git clone --recursive https://github.com/InfiniTensor/InfiniCore.git
cd InfiniCore/third_party
git clone https://github.com/Dao-AILab/flash-attention.git
git clone https://github.com/NVIDIA/cutlass.git
cd flash-attention/
git checkout 10846960ca0793b993446f6dbaf696479c127a9d
git checkout fbe15683a881743c2625b931cc4abcc107b43154
cd ../cutlass/
git checkout 087c84df83d254b5fb295a7a408f1a1d554085cf
git checkout e5fcd125a5cc1bd3e3a0063373e343b9beab6b14
cd ../../
#编译前注意注释掉cutlass_int8_scaled_mm_sm90函数中的内容，否则编译不通过

export CUTLASS_ROOT=~/code/2605server/static/InfiniCore/third_party/cutlass
export LD_LIBRARY_PATH=/home/mayuhang/env/.venv/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/home/mayuhang/env/.venv/include/python3.13:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/home/mayuhang/env/.venv/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/mayuhang/env/.venv/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH


xmake f --nv-gpu=y --ccl=y --cuda=$CUDA_HOME --aten=y --flash-attn=/home/mayuhang/code/2605server/static/InfiniCore/third_party/flash-attention -cv

xmake build && xmake install
xmake build _infinicore
xmake install _infinicore
pip install -e .
3. fla+graph
source /data/shared/spack/share/spack/setup-env.sh
spack load cmake@4.0.0
spack load llvm@16.0.6
spack load cuda@12.9.0
spack load cudnn@9.8.0.87-12
spack load nccl@2.27.5-1
spack find --loaded
source /home/mayuhang/env/.venv/bin/activate


git clone --recursive https://github.com/InfiniTensor/InfiniCore.git
cd InfiniCore/third_party
git clone https://github.com/Dao-AILab/flash-attention.git
git clone https://github.com/NVIDIA/cutlass.git
cd flash-attention/
git checkout 10846960ca0793b993446f6dbaf696479c127a9d
git checkout fbe15683a881743c2625b931cc4abcc107b43154
cd ../cutlass/
git checkout 087c84df83d254b5fb295a7a408f1a1d554085cf
git checkout e5fcd125a5cc1bd3e3a0063373e343b9beab6b14
cd ../../

export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export CUTLASS_ROOT=~/code/2605server/static/InfiniCore/third_party/cutlass
export LD_LIBRARY_PATH=/home/mayuhang/env/.venv/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/home/mayuhang/env/.venv/include/python3.13:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/home/mayuhang/env/.venv/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/mayuhang/env/.venv/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH


xmake f --nv-gpu=y --ccl=y --cuda=$CUDA_HOME --aten=y --graph=y --flash-attn=/home/mayuhang/code/2605server/static/InfiniCore/third_party/flash-attention -cv

xmake build && xmake install
xmake build _infinicore
xmake install _infinicore
pip install -e .
InfiniLM
编译
unset https_proxy
unset http_proxy
unset wss_proxy
unset all_proxy
export INFINI_ROOT="$HOME/.infini"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"
source /home/mayuhang/env/.venv/bin/activate
source /data/shared/spack/share/spack/setup-env.sh
spack load cmake@4.0.0
spack load llvm@16.0.6
spack load cuda@12.9.0
spack load cudnn@9.8.0.87-12
spack load nccl@2.27.5-1

xmake
xmake install
pip install -e .
1. 单节点服务--基础
export INFINI_ROOT="$HOME/.infini"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"
CUDA_VISIBLE_DEVICES=1 python python/infinilm/server/inference_server.py \
--device nvidia \
--model /data-aisoft/mechdancer/models/9g_8b_thinking \
--max-new-tokens=1024 \
--max-batch-size=16 \
--num-blocks=512 \
--tp=1 \
--temperature=1.0 \
--top-p=0.8 \
--top-k=1 \
--port=8001 \
--attn paged-attn \
--enable-paged-attn

7B：
--model /data-aisoft/mechdancer/models/9G7B_MHA \
8B：
--model /data-aisoft/mechdancer/models/9g_8b_thinking \
2. 单节点服务--fla
export LD_LIBRARY_PATH=/home/mayuhang/env/.venv/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=1 python python/infinilm/server/inference_server.py \
--device nvidia \
--model /data-aisoft/mechdancer/models/9g_8b_thinking \
--max-new-tokens=1024 \
--max-batch-size=32 \
--num-blocks=512 \
--tp=1 \
--temperature=1.0 \
--top-p=0.8 \
--top-k=1 \
--port=8001 \
--attn flash-attn \
--enable-paged-attn

7B：
--model /data-aisoft/mechdancer/models/9G7B_MHA \
8B：
--model /data-aisoft/mechdancer/models/9g_8b_thinking \
3. 单节点服务--fla+graph
export LD_LIBRARY_PATH=/home/mayuhang/env/.venv/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=1 python python/infinilm/server/inference_server.py \
--device nvidia \
--model /data-aisoft/mechdancer/models/9g_8b_thinking \
--max-new-tokens=1024 \
--max-batch-size=32 \
--num-blocks=512 \
--tp=1 \
--temperature=1.0 \
--top-p=0.8 \
--top-k=1 \
--port=8001 \
--attn flash-attn \
--enable-graph \
--enable-paged-attn

7B：
--model /data-aisoft/mechdancer/models/9G7B_MHA \
8B：
--model /data-aisoft/mechdancer/models/9g_8b_thinking \
MX
1. MX InfiniCore 编译
git clone --recursive https://github.com/InfiniTensor/InfiniCore.git

export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/opt/maca-3.2.1/tools/cu-bridge/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/opt/maca-3.2.1/include/:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/opt/maca-3.2.1/include/mcr:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/opt/maca-3.2.1/include/common:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/opt/maca-3.2.1/include/mcsparse:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/opt/maca-3.2.1/include/mcblas:$CPLUS_INCLUDE_PATH 
export CPLUS_INCLUDE_PATH=/opt/maca-3.2.1/include/mcsolver:$CPLUS_INCLUDE_PATH
export FLASH_ATTN_2_CUDA_SO=/opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda*.so

xmake f --metax-gpu=true --use-mc=true --ccl=true --graph=y --aten=y --flash-attn=. -cv

xmake build && xmake install
xmake build _infinicore && xmake install _infinicore
pip install -e . 
2. MX InfiniLM 编译 + 单节点服务测试
xmake build  _infinilm
xmake install  _infinilm
pip install -e .

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/python3.12:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

unset http_proxy https_proxy all_proxy ALL_PROXY

export MACA_VISIBLE_DEVICES=1
python python/infinilm/server/inference_server.py \
--device metax \
--model /data-aisoft/mechdancer/models/9g_8b_thinking \
--temperature 1.0 \
--top-p 0.8 \
--top-k 1 \
--port 8001 \
--tp 1  \
--block-size 256 \
--max-new-tokens 100 \
--num-blocks 2048 \
--max-batch-size 64 \
--enable-graph \
--enable-paged-attn \
--attn flash-attn \
--log-level INFO
