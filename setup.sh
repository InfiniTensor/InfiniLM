sudo apt-get update
sudo apt-get install cmake
sudo apt-get install gdb

wget https://xmake.io/shget.text -O - | bash
source ~/.xmake/profile
sudo sh cuda_12.0.0_525.60.13_linux.run
python InfiniCore/scripts/install.py --nv-gpu=y
sudo cp /environment/miniconda3/lib/python3.11/site-packages/nvidia/cudnn/include/cudnn*.h /usr/include
sudo cp /environment/miniconda3/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn* /usr/lib
sudo ln -s /usr/lib/libcudnn.so.8 /usr/lib/libcudnn.so
source ./start.sh
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
CUDA12.0 in local environment
export INFINI_ROOT=/home/featurize/.infini/