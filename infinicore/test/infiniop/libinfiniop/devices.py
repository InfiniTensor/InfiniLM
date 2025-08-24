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


InfiniDeviceNames = {
    InfiniDeviceEnum.CPU: "CPU",
    InfiniDeviceEnum.NVIDIA: "NVIDIA",
    InfiniDeviceEnum.CAMBRICON: "Cambricon",
    InfiniDeviceEnum.ASCEND: "Ascend",
    InfiniDeviceEnum.METAX: "Metax",
    InfiniDeviceEnum.MOORE: "Moore",
    InfiniDeviceEnum.ILUVATAR: "Iluvatar",
    InfiniDeviceEnum.KUNLUN: "Kunlun",
    InfiniDeviceEnum.SUGON: "Sugon",
}

# Mapping that maps InfiniDeviceEnum to torch device string
torch_device_map = {
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
