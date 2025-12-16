import argparse
from ctypes import POINTER, c_float, c_int, c_uint

import torch

from libinfinicore_infer import DeviceType, JiugeModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    # Load model via existing helper (loads weights + creates C++ model instance)
    from jiuge import JiugeForCauslLM

    llm = JiugeForCauslLM(
        args.model_dir, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=args.max_tokens
    )

    model: JiugeModel = llm.jiuge_model
    handle = llm.model_instance
    meta = llm.meta

    # Deterministic greedy sampling
    temperature = (c_float * 1)(1.0)
    topk = (c_uint * 1)(1)
    topp = (c_float * 1)(1.0)

    # One request with a short prompt
    tokens_t = torch.tensor(
        [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198],
        dtype=torch.int32,
    )
    tokens = tokens_t.numpy().astype("uint32")
    ntok = int(tokens_t.numel())
    tokens_c = (c_uint * ntok)(*tokens.tolist())

    req_lens = (c_uint * 1)(ntok)
    req_pos = (c_uint * 1)(0)

    dev_ids = (c_int * 1)(0)

    # Create KV caches explicitly
    kv0 = model.create_kv_cache(
        meta.nlayer,
        meta.dctx,
        meta.nkvh,
        meta.dh,
        meta.dh,
        meta.dt_logits,
        DeviceType.DEVICE_TYPE_CPU,
        dev_ids,
        1,
    )
    kv1 = model.create_kv_cache(
        meta.nlayer,
        meta.dctx,
        meta.nkvh,
        meta.dh,
        meta.dh,
        meta.dt_logits,
        DeviceType.DEVICE_TYPE_CPU,
        dev_ids,
        1,
    )

    from libinfinicore_infer import KVCacheCStruct

    kv_caches0 = (POINTER(KVCacheCStruct) * 1)(kv0)
    kv_caches1 = (POINTER(KVCacheCStruct) * 1)(kv1)

    out0 = (c_uint * 1)()
    model.infer_batch(
        handle,
        tokens_c,
        ntok,
        req_lens,
        1,
        req_pos,
        kv_caches0,
        temperature,
        topk,
        topp,
        out0,
    )

    # Build overrides equal to original embeddings for a few positions.
    emb = llm.weights.input_embd_tensor  # [dvoc, d], dtype == dt_logits
    d = int(meta.d)
    override_pos_list = [0, 3, ntok - 1]
    override_pos = (c_uint * len(override_pos_list))(*override_pos_list)

    override_embeds = torch.empty((len(override_pos_list), d), dtype=emb.dtype)
    for j, p in enumerate(override_pos_list):
        override_embeds[j].copy_(emb[int(tokens_t[p].item())])

    out1 = (c_uint * 1)()
    model.infer_batch_with_overrides(
        handle,
        tokens_c,
        ntok,
        req_lens,
        1,
        req_pos,
        kv_caches1,
        len(override_pos_list),
        override_pos,
        override_embeds.data_ptr(),
        temperature,
        topk,
        topp,
        out1,
    )

    print("jiuge override smoke:")
    print("  out_no_override:", int(out0[0]))
    print("  out_with_override:", int(out1[0]))

    model.drop_kv_cache(kv0)
    model.drop_kv_cache(kv1)
    model.destroy_model(handle)


if __name__ == "__main__":
    main()
