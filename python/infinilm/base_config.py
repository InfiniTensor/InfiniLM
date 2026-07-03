import argparse
import importlib
import json
import os
import shutil
import warnings

from infinilm.moe_config import MOE_EP_BACKEND_HELP


def parse_list(value: str):
    """Parse parse_list argument: can be a single int or a list of ints.

    Examples:
        "1" -> 1
        "[1,2,4]" -> [1, 2, 4]
        "1,2,4" -> [1, 2, 4]
    """
    value = value.strip()
    # Try to parse as JSON list first
    if value.startswith("[") and value.endswith("]"):
        try:
            result = json.loads(value)
            if isinstance(result, list):
                return [int(x) for x in result]
            return int(result)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to parse as comma-separated values
    if "," in value:
        try:
            return [int(x.strip()) for x in value.split(",")]
        except ValueError:
            pass

    # Try to parse as a single integer
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"value must be an int or list[int], got: {value}"
        )


class BaseConfig:
    """InfiniLM Unified Config - Command line argument parser"""

    def __init__(self):

        self.parser = argparse.ArgumentParser(description="InfiniLM Unified Config")
        self._add_common_args()
        self.args, self.extra = self.parser.parse_known_args()

        if self.extra:
            warnings.warn(
                f"Unrecognized arguments: {self.extra}. These arguments are not defined in BaseConfig.",
                UserWarning,
            )

        self.model = self.args.model
        self.device = self.args.device
        self.tp = self.args.tp
        self.dp = self.args.dp
        self.ep = self.args.ep
        self.moe_ep_backend = self.args.moe_ep_backend
        self.skip_legacy_moe = self.args.skip_legacy_moe

        self.attn = self.args.attn
        self.enable_graph = self.args.enable_graph
        self.enable_paged_attn = self.args.enable_paged_attn
        self.use_mla = self.args.use_mla
        self.num_blocks = self.args.num_blocks
        self.block_size = self.args.block_size
        self.max_cache_len = self.args.max_cache_len
        self.kv_cache_dtype = self.args.kv_cache_dtype
        self.skip_load = self.args.skip_load
        self.weight_load_mode = self.args.weight_load_mode

        self.batch_size = self.args.batch_size
        self.max_batch_size = self.args.max_batch_size
        self.input_len = self.args.input_len
        self.output_len = self.args.output_len
        self.max_new_tokens = self.args.max_new_tokens
        self.prompt = self.args.prompt
        self.top_k = self.args.top_k
        self.top_p = self.args.top_p
        self.temperature = self.args.temperature

        self.warmup = self.args.warmup
        self.verbose = self.args.verbose
        self.log_level = self.args.log_level

        # Evaluation parameters
        self.bench = self.args.bench
        self.backend = self.args.backend
        self.subject = self.args.subject
        self.split = self.args.split
        self.num_samples = self.args.num_samples
        self.output_csv = self.args.output_csv
        self.cache_dir = self.args.cache_dir

        # Quantization parameters
        self.awq = self.args.awq
        self.gptq = self.args.gptq
        self.dtype = self.args.dtype

        # Server parameters
        self.host = self.args.host
        self.port = self.args.port
        self.endpoint = self.args.endpoint
        self.ignore_eos = self.args.ignore_eos
        # PD separation (KV transfer)
        self.kv_transfer_config = self.args.kv_transfer_config

        # Multimodal parameters
        self.image = self.args.image
        self.video = self.args.video
        self.image_max_pixels = self.args.image_max_pixels
        self.image_min_pixels = self.args.image_min_pixels
        self.video_num_frames = self.args.video_num_frames
        self.video_max_pixels = self.args.video_max_pixels
        self.video_min_pixels = self.args.video_min_pixels
        self.skip_output = self.args.skip_output

        if self.image_max_pixels is not None:
            os.environ["INFINILM_VIDEONSA_IMAGE_MAX_PIXELS"] = str(
                self.image_max_pixels
            )
        if self.image_min_pixels is not None:
            os.environ["INFINILM_VIDEONSA_IMAGE_MIN_PIXELS"] = str(
                self.image_min_pixels
            )
        if self.video_num_frames is not None:
            os.environ["INFINILM_VIDEONSA_VIDEO_NUM_FRAMES"] = str(
                self.video_num_frames
            )
        if self.video_max_pixels is not None:
            os.environ["INFINILM_VIDEONSA_VIDEO_MAX_PIXELS"] = str(
                self.video_max_pixels
            )
        if self.video_min_pixels is not None:
            os.environ["INFINILM_VIDEONSA_VIDEO_MIN_PIXELS"] = str(
                self.video_min_pixels
            )

        if self.enable_paged_attn and self.attn == "default":
            self.attn = "paged-attn"

        # Force sync weight loading for Metax devices
        self._force_sync_for_metax()

    def _force_sync_for_metax(self):
        """Force weight_load_mode to 'sync' for Metax devices."""
        # Check if device is explicitly set to Metax
        if self.device.lower() == "metax":
            self.weight_load_mode = "sync"
            warnings.warn(
                "Metax device detected: forcing weight_load_mode to 'sync'",
                UserWarning,
            )
            return

        # Check if auto-detected device is Metax
        if self.device.lower() == "auto":
            detected_device = self.detect_device()
            if detected_device.lower() == "metax":
                self.weight_load_mode = "sync"
                warnings.warn(
                    "Auto-detected Metax device: forcing weight_load_mode to 'sync'",
                    UserWarning,
                )

    def _add_common_args(self):
        # --- base configuration ---
        self.parser.add_argument("--model", type=str, required=True)
        self.parser.add_argument(
            "--device",
            type=str,
            default="auto",
            help=(
                "device platform: auto, cpu, nvidia, qy, metax, moore, iluvatar, "
                "ali, cambricon, ascend, kunlun, hygon, or backend name "
                "(cuda/mlu/musa/npu)"
            ),
        )
        self.parser.add_argument("--tp", "--tensor-parallel-size", type=int, default=1)
        self.parser.add_argument("--dp", "--data-parallel-size", type=int, default=1)
        self.parser.add_argument(
            "--ep", "--expert-parallel-size", type=int, default=None
        )
        self.parser.add_argument(
            "--moe-ep-backend",
            type=str,
            default="auto",
            help=MOE_EP_BACKEND_HELP,
        )
        self.parser.add_argument(
            "--skip-legacy-moe",
            action="store_true",
            help="use the new fused MoE implementation instead of the legacy Qwen3 MoE MLP",
        )

        # --- Infer backend optimization ---
        self.parser.add_argument(
            "--attn",
            type=str,
            default="default",
            choices=["default", "paged-attn", "flash-attn"],
        )
        self.parser.add_argument("--enable-graph", action="store_true")
        self.parser.add_argument(
            "--use-mla",
            action="store_true",
            help="use DeepSeek V2 MLA attention when supported",
        )
        self.parser.add_argument(
            "--enable-paged-attn",
            action="store_true",
            help="use paged cache",
        )
        self.parser.add_argument(
            "--num-blocks", type=int, default=512, help="number of KV cache blocks"
        )
        self.parser.add_argument(
            "--block-size", type=int, default=256, help="size of each KV cache block"
        )
        self.parser.add_argument(
            "--max-cache-len", type=int, default=4096, help="maximum cache length"
        )
        self.parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            default=None,
            choices=["int8"],
            help="KV cache data type",
        )
        self.parser.add_argument(
            "--skip-load", action="store_true", help="skip loading model weights"
        )
        self.parser.add_argument(
            "--weight-load",
            dest="weight_load_mode",
            type=str,
            default="async",
            choices=["async", "sync"],
            help="weight loading mode across tensor-parallel workers",
        )

        # --- Length and infer parameters ---
        self.parser.add_argument("--batch-size", type=int, default=1)
        self.parser.add_argument(
            "--max-batch-size",
            type=int,
            default=8,
            help="maximum batch size for server",
        )
        self.parser.add_argument(
            "--input-len", type=parse_list, default=10, help="input sequence length"
        )
        self.parser.add_argument(
            "--output-len", type=parse_list, default=20, help="output sequence length"
        )
        self.parser.add_argument(
            "--max-new-tokens",
            type=int,
            default=512,
            help="maximum number of new tokens to generate",
        )
        self.parser.add_argument(
            "--prompt", type=str, default="How are you", help="default prompt text"
        )
        self.parser.add_argument("--top-k", type=int, default=1)
        self.parser.add_argument("--top-p", type=float, default=1.0)
        self.parser.add_argument("--temperature", type=float, default=1.0)

        # --- debug ---
        self.parser.add_argument("--warmup", action="store_true")
        self.parser.add_argument("--verbose", action="store_false")
        self.parser.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="logging level",
        )

        # --- Evaluation parameters ---
        self.parser.add_argument(
            "--bench",
            type=str,
            default=None,
            choices=["ceval", "mmlu"],
            help="benchmark to evaluate",
        )
        self.parser.add_argument(
            "--backend",
            type=str,
            default="cpp",
            choices=["python", "cpp", "torch", "vllm"],
            help="backend type",
        )

        self.parser.add_argument(
            "--subject",
            type=str,
            default="all",
            help="subject(s) to evaluate, comma-separated or 'all'",
        )
        self.parser.add_argument(
            "--split",
            type=str,
            default="test",
            choices=["test", "val", "all"],
            help="dataset split to use",
        )
        self.parser.add_argument(
            "--num-samples",
            type=int,
            default=None,
            help="number of samples to evaluate per subject",
        )
        self.parser.add_argument(
            "--output-csv",
            type=str,
            default=None,
            help="path to output CSV file for results",
        )
        self.parser.add_argument(
            "--cache-dir", type=str, default=None, help="directory for dataset cache"
        )

        # --- Quantization parameters ---
        self.parser.add_argument(
            "--awq", action="store_false", help="use AWQ quantization"
        )
        self.parser.add_argument(
            "--gptq", action="store_false", help="use GPTQ quantization"
        )
        self.parser.add_argument(
            "--dtype",
            type=str,
            default="float16",
            choices=["float32", "float16", "bfloat16"],
            help="data type for model",
        )

        # --- Server parameters ---
        self.parser.add_argument(
            "--host", type=str, default="0.0.0.0", help="server host"
        )
        self.parser.add_argument("--port", type=int, default=8000, help="server port")
        self.parser.add_argument(
            "--endpoint", type=str, default="/completions", help="API endpoint"
        )

        self.parser.add_argument(
            "--ignore-eos",
            action="store_true",
            dest="ignore_eos",
            default=False,
            help="Ignore EOS token and continue generation",
        )

        # --- Multimodal parameters ---
        self.parser.add_argument(
            "--image",
            type=str,
            default=None,
            help="image path for multimodal models",
        )
        self.parser.add_argument(
            "--image-max-pixels",
            type=int,
            default=200704,
            help="maximum image pixels for VideoNSA/Qwen-VL preprocessing",
        )
        self.parser.add_argument(
            "--image-min-pixels",
            type=int,
            default=None,
            help="minimum image pixels for VideoNSA/Qwen-VL preprocessing",
        )
        self.parser.add_argument(
            "--video",
            type=str,
            default=None,
            help="video path for multimodal models",
        )
        self.parser.add_argument(
            "--video-num-frames",
            type=int,
            default=None,
            help="number of frames for VideoNSA/Qwen-VL preprocessing",
        )
        self.parser.add_argument(
            "--video-max-pixels",
            type=int,
            default=200704,
            help="maximum video frame pixels for VideoNSA/Qwen-VL preprocessing",
        )
        self.parser.add_argument(
            "--video-min-pixels",
            type=int,
            default=None,
            help="minimum video frame pixels for VideoNSA/Qwen-VL preprocessing",
        )
        self.parser.add_argument(
            "--skip-output",
            action="store_true",
            help="skip printing sample prompt/output",
        )

        # ---- PD separation arguments ----
        self.parser.add_argument(
            "--kv-transfer-config",
            type=str,
            default=None,
            help=(
                "JSON object for KVTransferConfig. Allowed keys only: "
                "kv_connector, engine_id, kv_role, kv_connector_extra_config (omit any for defaults). "
                "Example: "
                '\'{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}\''
            ),
        )

    def _torch_device_available(self, device_type):
        module_name = {
            "cuda": None,
            "mlu": "torch_mlu",
            "musa": "torch_musa",
            "npu": "torch_npu",
        }.get(device_type)

        try:
            if module_name is not None:
                importlib.import_module(module_name)
            torch = importlib.import_module("torch")
        except Exception:
            return False

        device_module = getattr(torch, device_type, None)
        if device_module is None:
            return False

        is_available = getattr(device_module, "is_available", None)
        if not callable(is_available):
            return False

        try:
            return bool(is_available())
        except Exception:
            return False

    def detect_device(self):
        """Detect the local accelerator platform, falling back to CPU."""
        torch_checks = [
            ("cambricon", "mlu"),
            ("ascend", "npu"),
            ("moore", "musa"),
        ]
        for device_name, device_type in torch_checks:
            if self._torch_device_available(device_type):
                return device_name

        env_checks = [
            ("metax", ["MACA_PATH", "MACA_HOME", "MACA_ROOT"]),
            ("hygon", ["DTK_HOME", "DTK_PATH"]),
        ]
        for device_name, env_names in env_checks:
            if any(os.getenv(env_name) for env_name in env_names):
                return device_name

        command_checks = [
            ("cambricon", ["cnmon"]),
            ("ascend", ["npu-smi"]),
            ("moore", ["mthreads-gmi"]),
            ("metax", ["mx-smi", "ht-smi"]),
            ("hygon", ["hy-smi"]),
            ("ali", ["ppu-smi"]),
            ("iluvatar", ["ixsmi"]),
            ("nvidia", ["nvidia-smi"]),
        ]
        for device_name, commands in command_checks:
            if any(shutil.which(command) for command in commands):
                return device_name

        if self._torch_device_available("cuda"):
            return "nvidia"

        return "cpu"

    def get_device_str(self, device):
        """Convert device name to backend string (cuda/cpu/musa/mlu)"""
        DEVICE_STR_MAP = {
            "cpu": "cpu",
            "cuda": "cuda",
            "mlu": "mlu",
            "musa": "musa",
            "npu": "npu",
            "nvidia": "cuda",
            "qy": "cuda",
            "cambricon": "mlu",
            "ascend": "npu",
            "metax": "cuda",
            "moore": "musa",
            "iluvatar": "cuda",
            "kunlun": "kunlun",
            "hygon": "cuda",
            "ali": "cuda",
        }
        device = device.lower()
        if device == "auto":
            device = self.detect_device()
            print(f"Auto-detected device platform: {device}")
        return DEVICE_STR_MAP.get(device, "cpu")

    def __repr__(self):
        """String representation of configuration"""
        return f"BaseConfig(model='{self.model}', device='{self.device}', tp={self.tp})"
