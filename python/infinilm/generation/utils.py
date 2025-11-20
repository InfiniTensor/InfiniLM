import time
from typing import Optional
import infinicore
from ..cache_utils import Cache, DynamicCache
import numpy as np


def infini_to_ctype_dtype(infini_dtype):
    """Convert PyTorch data type to infinicore data type"""
    import ctypes

    if infini_dtype == infinicore.int32:
        return ctypes.c_int32
    elif infini_dtype == infinicore.float32:
        return ctypes.c_float
    else:
        raise ValueError(f"Unsupported py_dtype: {infini_dtype}")


def infini_to_numpy(infini_tensor: infinicore.Tensor):
    if infini_tensor.device.type != "cpu":
        infini_tensor_cpu = infini_tensor.to(infinicore.device("cpu", 0))
    else:
        infini_tensor_cpu = infini_tensor

    # 获取数据指针和形状信息
    data_ptr = infini_tensor_cpu.data_ptr()
    num_elements = infini_tensor_cpu.numel()
    original_shape = infini_tensor_cpu.shape

    # 创建1D NumPy数组（共享内存）
    ArrayType = infini_to_ctype_dtype(infini_tensor_cpu.dtype) * num_elements
    array = ArrayType.from_address(data_ptr)
    np_flat = np.ctypeslib.as_array(array)

    # 重塑为原始形状
    np_array = np_flat.reshape(original_shape)

    return np.copy(np_array)


infinicore.Tensor.to_numpy = infini_to_numpy


class GenerationMixin:
    def _get_initial_cache_position(
        self,
        bs: int,
        seq_length: int,
        device: infinicore.device,
    ) -> infinicore.Tensor:
        """Calculates `cache_position` for the pre-fill stage"""
        cache_position_list = [list(range(0, seq_length)) for i in range(bs)]

        return infinicore.from_list(
            cache_position_list, dtype=infinicore.int64, device=device
        )

    def prepare_inputs_for_generation(
        self,
        device: infinicore.device,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        """Prepare the model inputs for generation."""

        # 1. Handle BC:
        model_inputs = {}
        # -------------------------------------------------------------------- #
        #                 所需的: KV Cache
        # -------------------------------------------------------------------- #
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        # -------------------------------------------------------------------------- #
        #                     计算所需的，cache_position
        # -------------------------------------------------------------------------- #
        current_cache_position = kwargs.get("cache_position", None)
        if current_cache_position is None:
            # prill阶段
            bs, seq_len = kwargs["input_ids"].shape[0:2]
            model_inputs["cache_position"] = self._get_initial_cache_position(
                bs, seq_len, device
            )

        else:
            # decoder 阶段
            bs, seq_len = current_cache_position.shape
            last_position = current_cache_position.narrow(1, seq_len - 1, 1)

            one_value = infinicore.from_list(
                [1],
                dtype=last_position.dtype,
                device=last_position.device,
            ).view((bs, 1))

            next_position = one_value + last_position
            model_inputs["cache_position"] = next_position

        # -------------------------------------------------------------------- #
        #                 所需的: token的input_ids
        # -------------------------------------------------------------------- #
        if kwargs.get("next_token_id", None) is not None:
            next_token_id = kwargs["next_token_id"]
            model_inputs["input_ids"] = infinicore.from_list([[next_token_id]])

        # -------------------------------------------------------------------- #
        #                 其他
        # -------------------------------------------------------------------- #
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs

    def generate(
        self,
        input_ids: infinicore.Tensor,
        max_new_tokens: int,
        device: infinicore.device,
        tokenizer,
        config,
        **kwargs,
    ):
        model_kwargs = kwargs

        # -------------------------------------------------------------------- #
        #                       创建 cache                                      #
        # -------------------------------------------------------------------- #
        model_kwargs["use_cache"] = True
        model_kwargs["past_key_values"] = DynamicCache(config=self.config)

        # -------------------------------------------------------------------- #
        #                       _sample函数                                     #
        # -------------------------------------------------------------------- #
        result = self._sample(
            input_ids,
            max_new_tokens=max_new_tokens,
            device=device,
            tokenizer=tokenizer,
            config=config,
            **model_kwargs,
        )
        return result

    def _sample(
        self,
        input_ids: infinicore.Tensor,
        max_new_tokens: int,
        device: infinicore.device,
        tokenizer,
        config,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head.

        Parameters:
            input_ids (batch_size, seq_len): The sequence used as a prompt for the generation.
            max_new_tokens: Maximum number of new tokens.
            device: infinicore.device.
            tokenizer: translating data into raw text.
        """

        batch_size, seq_len = input_ids.shape[:2]

        eos_token_id = config.eos_token_id
        eos_token_id_list = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )

        # -------------------------------------------------------------------------- #
        #                     初始化 cache_position
        # -------------------------------------------------------------------------- #
        output_tokens_list = []

        model_kwargs["input_ids"] = input_ids
        model_kwargs["cache_position"] = None
        output_content = ""
        print()

        time_list = []
        for i in range(0, max_new_tokens):
            # -------------------------------------------------------------------------- #
            #                     prepare model inputs
            # -------------------------------------------------------------------------- #
            model_inputs = self.prepare_inputs_for_generation(device, **model_kwargs)

            model_kwargs["cache_position"] = model_inputs["cache_position"]

            # -------------------------------------------------------------------------- #
            #                     计算一次
            # -------------------------------------------------------------------------- #
            start_time = time.time()
            logits = self.forward(**model_inputs, return_dict=True)

            # -------------------------------------------------------------------------- #
            #                     处理输出
            # -------------------------------------------------------------------------- #
            token_scores = logits

            # -------------------------------------------------------------------------- #
            #                     random_sample
            # -------------------------------------------------------------------------- #
            batch_size, _, vocab_size = token_scores.shape

            next_tokens = infinicore.empty(
                (batch_size,),
                dtype=infinicore.int32,
                device=token_scores.device,
            )
            for i in range(0, batch_size):
                score = token_scores.narrow(0, i, 1).view([vocab_size])
                out = next_tokens.narrow(0, i, 1).view([])
                infinicore.nn.functional.random_sample(
                    score,
                    0.8,
                    0.1,
                    1,
                    1.0,
                    out=out,
                )
            end_time = time.time()
            time_list.append((end_time - start_time) * 1000)

            # ----------------------------------------------------------------- #
            #                得到下一个token的id，并解码为字符
            # ----------------------------------------------------------------- #
            token_id = next_tokens.to_numpy()[0]
            output_str = tokenizer.decode([token_id], skip_special_tokens=True)

            model_kwargs["next_token_id"] = token_id
            output_tokens_list.append(token_id)
            output_content += output_str

            print(output_str, end="", flush=True)
            if token_id in eos_token_id_list:
                break

        print(
            f"\n\n Time per step: {round(sum(time_list) / len(time_list), 2)} ms\n",
        )

        return output_tokens_list, output_content
