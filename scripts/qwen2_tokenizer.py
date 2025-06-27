# -*- coding: utf-8 -*-
"""
用法：
  python qwen2_tokenizer.py                         # 默认覆盖 tokenizer.json 文件
  python qwen2_tokenizer.py input.json output.json  # 指定输入输出路径
默认将覆盖原始文件 tokenizer.json,建议先备份。
"""
import json
import sys
import re

# tokenization_qwen2.py 的 bytes_to_unicode
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

_byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

def piece_to_text(piece: str) -> str:
    ba = bytearray()
    for ch in piece:
        b = _byte_decoder.get(ch)
        if b is not None:
            ba.append(b)
    text = ba.decode('utf-8', errors='replace')
    if not text or all(ord(c) < 32 or ord(c) == 127 for c in text):
        return ''
    return text

def check_missing_vocab(original_path="tokenizer.json", converted_path="tokenizer.json"):
    with open(original_path, "r", encoding="utf-8") as f:
        original = json.load(f)["model"]["vocab"]
    with open(converted_path, "r", encoding="utf-8") as f:
        converted = json.load(f)["model"]["vocab"]

    original_ids = set(original.values())
    converted_ids = set(converted.values())

    missing_ids = sorted(original_ids - converted_ids)
    missing_tokens = [(tok, idx) for tok, idx in original.items() if idx in missing_ids]

    print(f"🔎 原始 vocab 总数: {len(original)}")
    print(f"✅ 转换后 vocab 总数: {len(converted)}")
    print(f"⚠️ 缺失的 ID 数量: {len(missing_ids)}")
    if missing_tokens:
        print("示例缺失项 (前10项):")
        for tok, idx in missing_tokens[:10]:
            print(f"  ID={idx} -> token='{tok}'")

    return missing_tokens

def main(in_path='tokenizer.json', out_path=None):
    out_path = out_path or in_path

    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    orig_vocab = data['model']['vocab']

    id_to_piece = {idx: tok for tok, idx in orig_vocab.items()}
    sorted_ids = sorted(id_to_piece.keys())

    # 处理 vocab
    new_vocab = {}
    id_to_new_tok = {}
    for idx in sorted_ids:
        tok_str = id_to_piece[idx]
        decoded = piece_to_text(tok_str)
        if not decoded or '�' in decoded:
            real = tok_str
        else:
            real = decoded
        id_to_new_tok[idx] = real
        # 暂时不修正键值反转的情况，RTL字符待解决
        # if isinstance(real, str) and real.isdigit() and tok_str != real:
        #     try:
        #         swapped_idx = int(real)
        #         real, idx = tok_str, swapped_idx
        #     except ValueError:
        #         real = tok_str
        # elif isinstance(tok_str, str) and re.fullmatch(r'[0-9]+', tok_str):
        #     try:
        #         swapped_idx = int(tok_str)
        #         real, idx = real, swapped_idx
        #     except ValueError:
        #         pass

        if real not in new_vocab:
            new_vocab[real] = idx
        else:
            new_vocab[tok_str] = idx
            # print(f"⚠️ 重复 token 映射冲突: {repr(real)} 被忽略，原为 ID={new_vocab[real]}，当前 ID={idx}")

    # 处理 merges
    merges = data['model'].get('merges', [])
    new_merges = []
    for m in merges:
        a, b = m.split(' ', 1)
        ida = orig_vocab.get(a); idb = orig_vocab.get(b)
        if ida is None or idb is None:
            continue
        ta = id_to_new_tok.get(ida, a)
        tb = id_to_new_tok.get(idb, b)
        new_merges.append(ta + ' ' + tb)

    data['model']['vocab'] = new_vocab
    data['model']['merges'] = new_merges
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ {in_path} 转换完成，写入 {out_path}")

if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
    # check_missing_vocab() # 如果需要检查缺失的 vocab 条目，可以取消注释此行
