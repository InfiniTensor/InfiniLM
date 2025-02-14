class InfiniDeviceEnum:
    CPU = 0
    NVIDIA = 1
    CAMBRICON = 2
    ASCEND = 3
    METAX = 4
    MOORE = 5
    ILUVATAR = 6
    KUNLUN = 7
    SUGON = 8


# Mapping that maps InfiniDeviceEnum to torch device string
infiniDeviceEnum_str_map = {
    InfiniDeviceEnum.CPU: "cpu",
    InfiniDeviceEnum.NVIDIA: "cuda",
    InfiniDeviceEnum.CAMBRICON: "mlu",
    InfiniDeviceEnum.ASCEND: "npu",
    InfiniDeviceEnum.METAX: "cuda",
    InfiniDeviceEnum.MOORE: "musa",
    InfiniDeviceEnum.ILUVATAR: "cuda",
    InfiniDeviceEnum.KUNLUN: "cuda",
    InfiniDeviceEnum.SUGON: "cuda",
}
