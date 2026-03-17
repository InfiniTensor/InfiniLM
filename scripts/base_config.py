import argparse
import sys
from libinfinicore_infer import DeviceType




class BaseTestConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="InfiniLM Unified Config")
        self._add_common_args()
        
        # 核心：使用 parse_known_args() 容忍脚本特有参数
        # args 存储解析好的命名空间，extra 存储未识别的参数
        self.args, self.extra = self.parser.parse_known_args()

        self.model_path = self.args.model_path
        self.ndev = self.args.ndev
        self.verbose = self.args.verbose
        
        self.device_name = self.args.device
        self.device_type = self._get_device_type(self.args.device)



    def _add_common_args(self):

        self.parser.add_argument("--device", type=str, default="cpu")
        self.parser.add_argument("--model_path", type=str, required=True)
        self.parser.add_argument("--ndev", type=int, default=1)
        self.parser.add_argument("--verbose", action="store_true")


    def _get_device_type(self, dev_str):
        DEVICE_TYPE_MAP = {
            "cpu": DeviceType.DEVICE_TYPE_CPU,
            "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
            "qy": DeviceType.DEVICE_TYPE_QY,
            "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
            "ascend": DeviceType.DEVICE_TYPE_ASCEND,
            "metax": DeviceType.DEVICE_TYPE_METAX,
            "moore": DeviceType.DEVICE_TYPE_MOORE,
            "iluvatar": DeviceType.DEVICE_TYPE_ILUVATAR,
            "kunlun": DeviceType.DEVICE_TYPE_KUNLUN,
            "hygon": DeviceType.DEVICE_TYPE_HYGON,
        }

        return DEVICE_TYPE_MAP.get(dev_str.lower(), DeviceType.DEVICE_TYPE_CPU)

if __name__ == '__main__':
    cfg = BaseTestConfig()
    print(cfg.model_path)
    print(cfg.ndev)