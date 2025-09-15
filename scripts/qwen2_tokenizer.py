# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for Qwen2."""

import json
import os
import unicodedata
from functools import lru_cache
from typing import Optional, Dict, List, Tuple, Union

import regex as re


class AddedToken:
    """Simple wrapper for added tokens with special properties"""

    def __init__(
        self, content, special=True, lstrip=False, rstrip=False, normalized=False
    ):
        self.content = content
        self.special = special
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.normalized = normalized

    def __str__(self):
        return self.content

    def __repr__(self):
        return f"AddedToken(content='{self.content}', special={self.special})"


class PreTrainedTokenizer:
    """Base class for pretrained tokenizers with minimal implementation"""

    def __init__(
        self,
        errors="replace",
        bos_token=None,
        eos_token=None,
        pad_token=None,
        unk_token=None,
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        self.errors = errors
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.split_special_tokens = split_special_tokens

        # Handle token conversion
        self.bos_token = (
            bos_token.content if isinstance(bos_token, AddedToken) else bos_token
        )
        self.eos_token = (
            eos_token.content if isinstance(eos_token, AddedToken) else eos_token
        )
        self.pad_token = (
            pad_token.content if isinstance(pad_token, AddedToken) else pad_token
        )
        self.unk_token = (
            unk_token.content if isinstance(unk_token, AddedToken) else unk_token
        )

        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

    def __call__(self, text, **kwargs):
        return self.encode(text, **kwargs)

    def encode(self, text, **kwargs):
        tokens = self.tokenize(text)
        return {"input_ids": self.convert_tokens_to_ids(tokens)}

    def tokenize(self, text, **kwargs):
        return self._tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=None,
        **kwargs,
    ):
        if clean_up_tokenization_spaces is None:
            clean_up_tokenization_spaces = self.clean_up_tokenization_spaces

        # 确保 token_ids 是列表形式
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif hasattr(token_ids, "tolist"):  # 处理 numpy 数组或 torch tensor
            token_ids = token_ids.tolist()

        tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )

        # 过滤掉 None 值
        tokens = [token for token in tokens if token is not None]

        if not tokens:
            return ""

        text = self.convert_tokens_to_string(tokens)

        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)

        return text

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            token = self._convert_id_to_token(ids)
            if token is None:
                return self.unk_token if not skip_special_tokens else None
            if skip_special_tokens and token in [
                self.bos_token,
                self.eos_token,
                self.pad_token,
                self.unk_token,
            ]:
                return None
            return token

        tokens = []
        for id in ids:
            token = self._convert_id_to_token(id)
            if token is None:
                if not skip_special_tokens:
                    tokens.append(self.unk_token)
                continue
            if skip_special_tokens and token in [
                self.bos_token,
                self.eos_token,
                self.pad_token,
                self.unk_token,
            ]:
                continue
            tokens.append(token)
        return tokens

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 过滤掉 None 值
        tokens = [token for token in tokens if token is not None]
        if not tokens:
            return ""

        text = "".join(tokens)
        try:
            text = bytearray([self.byte_decoder[c] for c in text]).decode(
                "utf-8", errors=self.errors
            )
        except KeyError as e:
            # 处理未知字符的情况
            print(f"Warning: Unknown character in byte_decoder: {e}")
            # 尝试直接返回文本
            pass
        return text

    def clean_up_tokenization(self, text):
        """Clean up tokenization artifacts"""
        text = text.replace(" .", ".").replace(" ?", "?").replace(" !", "!")
        text = text.replace(" ,", ",").replace(" ' ", "'").replace(" n't", "n't")
        text = text.replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
        text = text.replace(" 'm", "'m").replace(" 'll", "'ll").replace(" 'd", "'d")
        return text.strip()

    def _tokenize(self, text):
        raise NotImplementedError

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token is None:
            return self.encoder.get(self.unk_token, 0)
        return self.encoder.get(token, self.encoder.get(self.unk_token, 0))


@lru_cache(maxsize=None)
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class Qwen2Tokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        # Qwen vocab does not contain control tokens; added tokens need to be special
        bos_token = (
            AddedToken(
                bos_token, lstrip=False, rstrip=False, special=True, normalized=False
            )
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(
                eos_token, lstrip=False, rstrip=False, special=True, normalized=False
            )
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(
                unk_token, lstrip=False, rstrip=False, special=True, normalized=False
            )
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(
                pad_token, lstrip=False, rstrip=False, special=True, normalized=False
            )
            if isinstance(pad_token, str)
            else pad_token
        )

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = []
        with open(merges_file, encoding="utf-8") as merges_handle:
            for i, line in enumerate(merges_handle):
                line = line.strip()
                if (i == 0 and line.startswith("#version:")) or not line:
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        self.pat = re.compile(PRETOKENIZE_REGEX)

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            raise ValueError(
                f"Vocabulary path ({save_directory}) should be a directory"
            )

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        merge_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "merges.txt",
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            )

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(
                self.bpe_ranks.items(), key=lambda kv: kv[1]
            ):
                if index != token_index:
                    print(
                        f"Warning: BPE merge indices are not consecutive. Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, **kwargs):
        text = unicodedata.normalize("NFC", text)
        return (text, kwargs)


# Example usage:
if __name__ == "__main__":
    # Initialize tokenizer with your vocab files
    tokenizer = Qwen2Tokenizer(
        vocab_file="path/to/vocab.json", merges_file="path/to/merges.txt"
    )

    # Tokenize text
    text = "Hello world"
    tokens = tokenizer.tokenize(text)
    print("Tokens:", tokens)

    # Convert to IDs
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print("IDs:", ids)

    # Decode back to text
    decoded = tokenizer.decode(ids)
    print("Decoded:", decoded)
