import time
from abc import ABC, abstractmethod


def _apply_chat_template_or_fallback(renderer, conversation, add_generation_prompt=True):
    if hasattr(renderer, "apply_chat_template"):
        try:
            prompt = renderer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            if isinstance(prompt, str):
                return prompt
        except (ValueError, TypeError):
            pass

    tokenizer = getattr(renderer, "tokenizer", renderer)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    parts = []
    for message in conversation:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not content:
            continue
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(content)
    prompt = "\n".join(parts)
    if add_generation_prompt:
        prompt += "\n"
    return prompt


def render_ceval(renderer, conversation):
    return _apply_chat_template_or_fallback(renderer, conversation) + "答案："


def render_mmlu(renderer, question, choices):
    choices_text = "\n".join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    )
    instruction = (
        "You are a multiple-choice question solver. "
        "Select the correct option and respond with only the letter A, B, C, or D."
    )
    prompt = f"{instruction}\n\nQuestion: {question}\n{choices_text}\nAnswer:"

    conversation = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"{question}\n{choices_text}\n"},
    ]
    try:
        return _apply_chat_template_or_fallback(renderer, conversation) + "The answer is: "
    except Exception:
        return prompt


class BaseBenchmark(ABC):
    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.total_tokens = 0
        self.total_time = 0.0

    def encode_text(self, text):
        return self.tokenizer.encode(text)

    def decode_token(self, token_id):
        return self.tokenizer.decode(token_id)

    def max_context_len(self):
        return self.config_dict.get("max_position_embeddings", 2048)

    def render_input_content(self, *args, **kwargs):
        renderer = getattr(self, "processor", self.tokenizer)
        if self.benchmark == "ceval":
            return render_ceval(renderer, *args, **kwargs)
        if self.benchmark == "mmlu":
            return render_mmlu(renderer, *args, **kwargs)
        raise ValueError(f"Unknown benchmark: {self.benchmark}")

    def record_generation(self, output_text, input_tokens, new_tokens, start_time):
        elapsed = time.perf_counter() - start_time
        total_tokens = input_tokens + new_tokens
        throughput = total_tokens / elapsed if elapsed > 0 else 0.0

        print(output_text)
        print()
        print(f"Total time: {elapsed * 1000:.2f} ms")
        print(f"Input tokens: {input_tokens}")
        print(f"New tokens: {new_tokens}")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Throughput: {throughput:.2f} tok/s")

        self.total_tokens += total_tokens
        self.total_time += elapsed
        return output_text

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    @abstractmethod
    def destroy_model_instance(self):
        pass
