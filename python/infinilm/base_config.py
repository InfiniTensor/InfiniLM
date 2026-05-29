import argparse
import json
import os
import sys
import warnings


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
            f"batch-size must be an int or list[int], got: {value}"
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

        self.attn = self.args.attn
        self.enable_graph = self.args.enable_graph
        self.enable_chunk_prefill_graph = self.args.enable_chunk_prefill_graph
        self.chunk_size = self.args.chunk_size
        self.enable_paged_attn = self.args.enable_paged_attn
        self.num_blocks = self.args.num_blocks
        self.block_size = self.args.block_size
        self.max_cache_len = self.args.max_cache_len
        self.kv_cache_dtype = self.args.kv_cache_dtype
        self.skip_load = self.args.skip_load

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

        if self.enable_paged_attn and self.attn == "default":
            self.attn = "paged-attn"

    def _add_common_args(self):
        # --- base configuration ---
        self.parser.add_argument("--model", type=str, required=True)
        self.parser.add_argument("--device", type=str, default="cpu")
        self.parser.add_argument("--tp", "--tensor-parallel-size", type=int, default=1)

        # --- Infer backend optimization ---
        self.parser.add_argument(
            "--attn",
            type=str,
            default="default",
            choices=["default", "paged-attn", "flash-attn"],
        )
        self.parser.add_argument("--enable-graph", action="store_true")
        self.parser.add_argument("--enable-chunk-prefill-graph", action="store_true", help="enable chunk-prefill graph compiling")
        self.parser.add_argument("--chunk-size", type=int, default=0, help="tokens per chunked-prefill slice (0 to disable)")
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

    def get_device_str(self, device):
        """Convert device name to backend string (cuda/cpu/musa/mlu)"""
        DEVICE_STR_MAP = {
            "cpu": "cpu",
            "nvidia": "cuda",
            "qy": "cuda",
            "cambricon": "mlu",
            "ascend": "ascend",
            "metax": "cuda",
            "moore": "musa",
            "iluvatar": "cuda",
            "kunlun": "kunlun",
            "hygon": "cuda",
            "ali": "cuda",
        }
        return DEVICE_STR_MAP.get(device.lower(), "cpu")

    def __repr__(self):
        """String representation of configuration"""
        return f"BaseConfig(model='{self.model}', device='{self.device}', tp={self.tp})"
