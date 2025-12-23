import infinicore
from transformers import AutoTokenizer
from infinilm.modeling_utils import load_model_state_dict_by_file
import infinilm
from infinilm.distributed import DistConfig
import argparse
import sys
import time
import os
import json
from collections import OrderedDict
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))


DATA_TYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}

# BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]
# INPUT_LENS = [32, 256, 1024, 4096]
# OUTPUT_LENS = [256, 1024, 4096]


def read_json_file(file_path):
    """Load and return JSON content from file_path."""
    with open(file_path, "r") as file:
        return json.load(file)


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


def get_test_cases(
    model_path: str,
    batch_size_list: list[int],
    input_len_list: list[int],
    output_len_list: list[int],
):
    model_path = os.path.expanduser(model_path)

    """Generate cases ordered by ascending KV cache memory usage."""
    # Load model config to derive attention dimensions
    config = read_json_file(os.path.join(model_path, "config.json"))
    head_dim = config.get(
        "head_dim", config.get("hidden_size") // config.get("num_attention_heads")
    )
    # KV heads and layers drive cache size
    num_key_value_heads = config.get("num_key_value_heads")
    num_hidden_layers = config.get("num_hidden_layers")

    # Enumerate all batch/input/output combinations and compute KV cache size
    case_list = []
    for batch_size in batch_size_list:
        for input_len in input_len_list:
            for output_len in output_len_list:
                for data_type in ["bfloat16"]:
                    data_type_bytes = DATA_TYPE_BYTES[data_type]

                    total_seq_len = input_len + output_len
                    kvcache_memory_bytes = (
                        data_type_bytes
                        * (batch_size * total_seq_len * num_key_value_heads * head_dim)
                        * num_hidden_layers
                    )
                    kvcache_memory_gb = kvcache_memory_bytes / (1024 * 1024 * 1024)

                    case_list.append(
                        {
                            "idx": len(case_list),
                            "batch_size": batch_size,
                            "input_len": input_len,
                            "output_len": output_len,
                            "data_type": data_type,
                            "kvcache_memory": round(kvcache_memory_gb, 3),
                        }
                    )

    # Sort by KV cache size and wrap in OrderedDict with index keys
    case_dict = OrderedDict(
        (idx, case)
        for idx, case in enumerate(
            sorted(case_list, key=lambda case: case["kvcache_memory"])
        )
    )

    return case_dict


def get_args():
    parser = argparse.ArgumentParser(description="run Llama args")

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run cpu test",
    )
    parser.add_argument(
        "--nvidia",
        action="store_true",
        help="Run nvidia test",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model path",
    )
    parser.add_argument(
        "--batch-size",
        type=parse_list,
        default=1,
        help="number of prompts in a batch (can be an int or a list of ints, e.g., '1' or '[1,2,4]' or '1,2,4')",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=1,
        help="total rank for tensor parallel",
    )
    parser.add_argument(
        "--input-len",
        type=parse_list,
        default=10,
        help="output tokens",
    )

    parser.add_argument(
        "--output-len",
        type=parse_list,
        default=20,
        help="output tokens",
    )
    return parser.parse_args()


prompt = "泰山，又名岱山、岱宗、岱岳、东岳、泰岳，为五岳之一，有“五岳之首”、“五岳独尊”、“天下第一山”、“华夏神山”之称 ，被中外学者称为“中国的奥林匹斯山” 位于山东省中部，隶属于泰安市，绵亘于泰安、济南、淄博三市之间，总面积25000公顷，主峰玉皇顶海拔约1545米。泰山相伴上下五千年的华夏文明传承历史，集国家兴盛、民族存亡的象征于一身，是中华民族的精神家园 [31]，东方文化的缩影，“天人合一”思想的寄托之地 [24]，承载着丰厚的地理历史文化内涵 [15]，被古人视为“直通帝座”的天堂，成为百姓崇拜，帝王告祭的神山，有“泰山安，四海皆安”的说法 [1]。自秦始皇起至清代，先后有13代帝王亲登泰山封禅或祭祀，另有24代帝王遣官祭祀72次。山体上既有寺庙、宫、观等古建筑群29处，古遗址128处，有大小碑碣、摩崖石刻2000余处 [15]。其景巍峨雄奇、幽奥俊秀，有石坞松涛、云海玉盘等美丽壮阔的自然景观。其历史文化、自然风光、地质奇观和谐融为一体，具有特殊的历史、文化、美学和科学价值。 [19]1982年，泰山被列入第一批国家级风景名胜区。1987年，泰山被联合国教科文组织批准列为全球首例世界文化与自然双重遗产 [14] [41-42]。2002年，泰山被评为“中华十大文化名山”之首 [15]。2005年，泰山成为国家地质公园。2006年，泰山因其独特的地质价值成为世界地质公园 [14]。2007年3月，泰山被评为国家AAAAA级旅游景区；12月，泰山被命名为中国首座“中国书法名山”。2025年3月20日，泰山迎来2025年第100万名游客。"


def repeat_prompt(input_ids: list[int], target_length: int):
    num = len(input_ids)
    repeat_times = (target_length + num - 1) // num
    return (input_ids * repeat_times)[:target_length]


class TestModel:
    model: infinicore.nn.Module
    tokenizer: AutoTokenizer
    input_ids_list: list[int]

    def __init__(
        self,
        model_path,
        infini_device=infinicore.device("cpu", 0),
        tp=1,
    ) -> None:
        model_path = os.path.expanduser(model_path)
        # ---------------------------------------------------------------------------- #
        #                        创建模型,
        # ---------------------------------------------------------------------------- #
        model = infinilm.AutoLlamaModel.from_pretrained(
            model_path,
            device=infini_device,
            backend="cpp",
            distributed_config=DistConfig(tp),
        )

        # ---------------------------------------------------------------------------- #
        #                        加载权重
        # ---------------------------------------------------------------------------- #
        load_model_state_dict_by_file(model, model_path, dtype=model.config.dtype)

        # ---------------------------------------------------------------------------- #
        #                        创建 tokenizer
        # ---------------------------------------------------------------------------- #
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ---------------------------------------------------------------------------- #
        #                        token编码
        # ---------------------------------------------------------------------------- #
        input_content = [
            tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        ]

        # print(input_content, end="", flush=True)
        input_ids_list = tokenizer.batch_encode_plus(input_content)["input_ids"]

        self.model = model
        self.tokenizer = tokenizer
        self.input_ids_list = input_ids_list

    def run(
        self,
        batch_size: int,
        input_len: int,
        output_len: int,
    ):
        input_ids = repeat_prompt(self.input_ids_list[0], target_length=input_len)
        input_ids_list = [input_ids] * batch_size

        # ---------------------------------------------------------------------------- #
        #                        自回归生成
        # ---------------------------------------------------------------------------- #
        input_ids_infini = infinicore.from_list(input_ids_list)

        t1 = time.time()
        print("=================== start generate ====================")
        self.model.generate(
            input_ids_infini,
            max_new_tokens=output_len,
            tokenizer=self.tokenizer,
            stop_on_eos=False,
        )
        t2 = time.time()

        print(
            f"total_time: {round((t2 - t1) * 1000, 2)} ms",
        )


if __name__ == "__main__":
    args = get_args()
    print(args)

    # Parse command line arguments
    device_str = "cpu"
    if args.cpu:
        device_str = "cpu"
    elif args.nvidia:
        device_str = "cuda"
    else:
        print(
            "python examples/bench.py --nvidia --model=~/TinyLlama-1.1B-Chat-v1.0/ --batch-size=2 --tp=1 --input-len=50 --output-len=50"
        )
        sys.exit(1)
    # -------------------------------------------------------- #
    #             解析参数
    # -------------------------------------------------------- #
    model_path = args.model

    infini_device = infinicore.device(device_str, 0)

    tp = args.tensor_parallel_size

    batch_size = args.batch_size
    input_len = args.input_len
    output_len = args.output_len

    if isinstance(batch_size, int):
        batch_size = [batch_size]

    if isinstance(input_len, int):
        input_len = [input_len]

    if isinstance(output_len, int):
        output_len = [output_len]

    cases_dict = get_test_cases(model_path, batch_size, input_len, output_len)
    # -------------------------------------------------------- #
    #             测试
    # -------------------------------------------------------- #
    # print("=================== start test ====================", type(batch_size))

    test = TestModel(
        model_path,
        infini_device=infini_device,
        tp=tp,
    )

    for idx, case in tqdm(cases_dict.items(), desc="Processing cases"):
        tqdm.write(f"\033[92mProcessing : {case}\033[0m")

        batch_size = case["batch_size"]
        input_len = case["input_len"]
        output_len = case["output_len"]

        # reset cache for each case
        initial_capacity = input_len + output_len + 100
        test.model.reset_cache(
            batch_size=batch_size, pos=0, initial_capacity=initial_capacity
        )

        # run test one case
        test.run(
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
        )
