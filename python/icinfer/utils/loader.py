import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# def load_model(model: nn.Module, path: str):
#     packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
#     for file in glob(os.path.join(path, "*.safetensors")):
#         with safe_open(file, "pt", "cpu") as f:
#             for weight_name in f.keys():
#                 for k in packed_modules_mapping:
#                     if k in weight_name:
#                         v, shard_id = packed_modules_mapping[k]
#                         param_name = weight_name.replace(k, v)
#                         param = model.get_parameter(param_name)
#                         weight_loader = getattr(param, "weight_loader")
#                         weight_loader(param, f.get_tensor(weight_name), shard_id)
#                         break
#                 else:
#                     param = model.get_parameter(weight_name)
#                     weight_loader = getattr(param, "weight_loader", default_weight_loader)
#                     weight_loader(param, f.get_tensor(weight_name))


def load_model(model: nn.Module, path: str):
    """
    智能加载模型权重。
    优先查找并使用 .safetensors 文件。如果找不到，则回退到查找并使用 .bin 文件。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 优先尝试加载 .safetensors 文件
    model_files = glob(os.path.join(path, "*.safetensors"))
    is_safetensors = True

    # 如果没有找到 .safetensors，则回退到加载 .bin 文件
    if not model_files:
        model_files = glob(os.path.join(path, "*.bin"))
        is_safetensors = False

    # 如果两种文件都找不到，则报错
    if not model_files:
        raise FileNotFoundError(f"No model weights found in {path}. Looked for .safetensors and .bin files.")

    # 核心加载逻辑
    for file_path in model_files:
        if is_safetensors:
            with safe_open(file_path, "pt", "cpu") as f:
                for weight_name in f.keys():
                    tensor = f.get_tensor(weight_name)
                    _load_and_dispatch(model, weight_name, tensor, packed_modules_mapping)
        else: # .bin format
            state_dict = torch.load(file_path, map_location="cpu")
            for weight_name, tensor in state_dict.items():
                _load_and_dispatch(model, weight_name, tensor, packed_modules_mapping)

def _load_and_dispatch(model, weight_name, tensor, packed_modules_mapping):
    """
    一个辅助函数，用于分派权重到正确的加载器。
    """
    is_packed = False
    for packed_key, (target_name, shard_id) in packed_modules_mapping.items():
        if packed_key in weight_name:
            # 替换名称以匹配模型中的合并层参数
            param_name = weight_name.replace(packed_key, target_name)
            try:
                param = model.get_parameter(param_name)
                # 调用参数上附加的专用加载器 (例如 QKVParallelLinear.weight_loader)
                getattr(param, "weight_loader")(param, tensor, shard_id)
            except AttributeError:
                print(f"Warning: Could not find parameter '{param_name}' for packed weight '{weight_name}'. Skipping.")
            is_packed = True
            break
    
    if not is_packed:
        try:
            param = model.get_parameter(weight_name)
            # 调用参数上附加的加载器，或使用默认加载器
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, tensor)
        except AttributeError:
            # 某些权重（如_pre_transformer_block.0.norm_1.weight）可能不存在于我们的模型中，可以安全地忽略
            pass



