import sys
import logging
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from icinfer.engine.libinfinicore_infer import DeviceType
from icinfer.models.jiuge import JiugeForCausalLM
logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/Llama-2-7b-chat-hf")
    # parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/FM9G_70B_SFT_MHA/")
    parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/9G7B_MHA/")
    parser.add_argument("--device-type", type=str, default="nvidia")
    parser.add_argument("--ndev", type=int, default=4)
    args = parser.parse_args()
    return args

def test():
    args = parse_args()
    model_path = args.model_path
    device_type = DeviceType.DEVICE_TYPE_CPU
    if args.device_type == "cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif args.device_type == "nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif args.device_type == "cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif args.device_type == "ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    elif args.device_type == "metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    elif args.device_type == "moore":
        device_type = DeviceType.DEVICE_TYPE_MOORE
    elif args.device_type == "iluvatar":
        device_type = DeviceType.DEVICE_TYPE_ILUVATAR
    else:
        logger.info(
            # "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
            "Usage: python jiuge.py [cpu | nvidia| cambricon | ascend | metax | moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = args.ndev
    model = JiugeForCausalLM(model_path, device_type, ndev)
    # model.generate(["山东最高的山是？", "中国面积最大的省是？"], 500)
    # model.generate(["山东最高的山是？"], 500)
    model.generate("山东最高的山是？", 500)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()
