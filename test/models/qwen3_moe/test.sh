# speed compare
srun --gres=gpu:nvidia:2 --cpus-per-task=16 --mem=256G python ourmoe_test.py --nvidia --model_path=/data/users/whitecity/models/Qwen3-30B-A3B-Instruct-2507-Layer-0
# result compare
srun --gres=gpu:nvidia:2 --cpus-per-task=16 --mem=256G python ourmoe_test.py --cpu --check --check_device cpu --seed 0 --model_path=/data/users/whitecity/models/Qwen3-30B-A3B-Instruct-2507-Layer-0 --layer_idx=0