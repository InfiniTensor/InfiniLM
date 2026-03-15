### MiniCPM‑SALA sanity alignment – current status

### Scope

- **Goal**: Align InfiniLM MiniCPM‑SALA logits with HF reference on the dense/GLA (non‑sparse) path, using the `examples/minicpm_sala_logits_sanity.py` script running inside the `minicpm-sala` container.

---

### Instrumentation and plumbing

- **Sanity script (`minicpm_sala_logits_sanity.py`)**
  - **Backend lock**: All InfiniLM `InferEngine` paths now use `attention_backend="default"` so they hit the dense/GLA fallback.
  - **Debug log target**: The script sets `INFINI_DEBUG_LOG=/home/zenghua/repos/.cursor/debug-9146ea.log` and `INFINI_DEBUG_ATTN_DUMP=1` so both Python and C++ write to the same NDJSON file.
  - **HF per-layer hooks**:
    - `_register_hf_layer_hooks` walks the model (`hf.transformer.layers`, `hf.model.layers`, or `hf.layers`) and registers forward hooks on the first 3 layers.
    - For each layer \(i\), it logs:
      - `min`, `max`, `mean`, `l2` of the layer output, as `hypothesisId="HF_L"`, `data.layer = i`.
    - Hooks are installed for `run_prefill_only` and removed after the forward pass.

- **InfiniLM attention (`minicpm_sala_attention.cpp`)**
  - Existing **layer‑0** diagnostics:
    - At entry to `forward_dense_`: `forward_dense_entry` logs env/config, including `INFINI_DEBUG_ATTN_DUMP`, `use_rope`, `use_qk_norm`, `use_output_gate`, `use_output_norm`, `is_sparse_layer`, and shapes.
    - For layer 0, logs stats for:
      - Pre‑gate attention output (`attn_pre_gate`): full tensor min/max/mean, `l2`, shape and scaling.
      - Post‑gate/norm (`attn_post_gate`), and post‑`o_proj` (`attn_post_oproj`).
  - **Planned / partially implemented**: extended logging for `layer_idx_ < 2` (layers 0 and 1) with:
    - `attn_pre_gate_l0` / `attn_pre_gate_l1`.
    - `attn_post_gate_l0` / `attn_post_gate_l1`.
    - `attn_post_oproj_l0` / `attn_post_oproj_l1`.
  - Current runs still only show layer‑0 entries; the `_infinilm` binary in use has not yet picked up the `_l1` variants (see below).

- **InfiniLM decoder layer (`minicpm_sala_decoder_layer.cpp/.hpp`)**
  - **MuP residual scaling**:
    - `residual_scale_ = scale_depth / sqrt(num_hidden_layers)` using `scale_depth` from `ModelConfig` (matches HF path).
    - `forward` applies:
      - `out1 = hidden_states + residual_scale_ * attn_out`.
      - `out2 = out1 + residual_scale_ * mlp_out`.
  - **Per-layer Inf output stats**:
    - New member `size_t layer_idx_` stored from constructor.
    - For `layer_idx_ < 3`, after computing `out2`, it:
      - Copies to CPU, converts BF16/F16/F32 to float, computes `min`, `max`, `mean`, `l2` and shape.
      - Logs as `hypothesisId="INF_L"`, with `data.layer = layer_idx_`.

- **Weight scaling / MuP configuration (`modeling_utils.py`)**
  - Loader reads `config.json` and applies MiniCPM‑style scaling:
    - `scale_input = scale_emb`, `scale_depth`, `num_hidden_layers`, `dim_model_base`, `hidden_size`.
    - For `model_type == "minicpm_sala"`:
      - `scale_o` and `scale_down` are reset to 1.0 (residual scaling is done at C++ forward time).
      - `scale_lm_head = dim_model_base / hidden_size` is baked into `lm_head.weight`.
    - Embedding and norm weights are scaled as in the MiniCPM scripts.

- **Rebuild and install (`rebuild.sh`, xmake)**
  - `rebuild.sh`:
    - `InfiniCore`: `python scripts/install.py --nv-gpu=y --ccl=y --aten=y`, then `xmake build _infinicore` and `xmake install _infinicore`.
    - `InfiniLM`: optional `xmake clean`, then `xmake build _infinilm` and `xmake install _infinilm`.
  - Verified inside container:
    - Shared libs in `/root/.infini/lib` are updated (e.g. `libinfiniop.so`, `libinfinicore_cpp_api.so` with current timestamps).
    - Python sees `infinilm` from `/home/zenghua/repos/InfiniLM/python/infinilm`.
    - The extension in use is `_infinilm` at:
      - `/home/zenghua/repos/InfiniLM/python/infinilm/lib/_infinilm.cpython-312-x86_64-linux-gnu.so`.

---

### Sanity run behavior and current misalignment

- **Command used (container, GPU 1)**:
  ```bash
  docker exec -e CUDA_VISIBLE_DEVICES=1 minicpm-sala bash -lc '
    source /app/docker/nvidia/env-set.sh
    cd /home/zenghua/repos/InfiniLM
    python3 examples/minicpm_sala_logits_sanity.py \
      --model_path /data-aisoft/zenghua/models/OpenBMB/MiniCPM-SALA \
      --mode prefill \
      --prompt "How are you"
  '
  ```
- **HF vs Inf logits (from `SANITY_ONELINE`)**
  - `inf_norm ≈ 387.66`
  - `hf_norm ≈ 1588.89`
  - **ratio_inf_hf ≈ 0.244**
  - `max_diff ≈ 12.77`, `mean_diff ≈ 4.64`
  - Top‑1 token IDs differ (HF: 74, Inf: 59358).

- **HF early layers (from `HF_L` logs)**
  - Using the HF hooks in the sanity script:
    - Layer 0: `l2 ≈ 59.49`
    - Layer 1: `l2 ≈ 73.91` (first GLA layer)
    - Layer 2: `l2 ≈ 87.38`
  - Norms grow smoothly with depth; nothing obviously pathological on HF side.

- **Inf attention layer‑0 vs HF**
  - HF layer‑0 pre‑gate attention (`modeling_minicpm_sala.py:attn_pre_gate`):
    - Shape `[1, 4, 4096]`, `min=-8.375`, `max=9.0`, `mean≈-0.1273`.
  - Inf layer‑0:
    - **Pre‑gate (`attn_pre_gate`)**:
      - `l2 ≈ 105.50`, `min=-8.375`, `max=9.0`.
      - Python’s comparison (`compare_attn`) reports `norm_ratio_inf_hf ≈ 0.4487`, i.e. Inf pre‑gate norm ≈ 0.45× HF’s.
    - **Post‑gate/norm (`attn_post_gate`)**:
      - `l2 ≈ 60.38`, very close to HF layer‑0 output `l2 ≈ 59.49`.
    - **Post‑o_proj (`attn_post_oproj`)**:
      - `l2 ≈ 98.66` (used as input to the decoder’s residual path).
  - Interpretation:
    - By the end of the **layer‑0 attention block**, Inf and HF are roughly matched in scale at the decoder output (norms ≈ 60).
    - The severe **0.244 logits norm ratio** is therefore not due to an immediate blow‑up/vanish at layer‑0 attention output; it accumulates later (likely starting at the first GLA layer and/or via MuP/residual/MLP scaling).

---

### Binary / build state

- **Extension module mapping**
  - In container, importing `infinilm` shows:
    - `infinilm.__file__` → `/home/zenghua/repos/InfiniLM/python/infinilm/__init__.py`
    - `_infinilm` (top‑level) → `/home/zenghua/repos/InfiniLM/python/infinilm/lib/_infinilm.cpython-312-x86_64-linux-gnu.so`
  - That is the `.so` used by the sanity script.

- **Why new attention logs for layer 1 don’t appear yet**
  - `strings _infinilm.cpython-312-...so | grep 'attn_pre_gate_l1'` currently returns **no matches**:
    - This confirms the loaded `_infinilm` was built **before** we added the `_l1` logging strings.
  - We attempted a fresh `_infinilm` build and initially hit:
    - C++ error in `MiniCPMSALADecoderLayer::forward`: `layer_idx_` not declared.
  - That prevented `_infinilm` from rebuilding/overwriting the old `.so`, so your layer‑1 logging changes never reached runtime.

- **Decoder fix applied to unblock rebuild**
  - Added `size_t layer_idx_ = 0;` as a private member in `minicpm_sala_decoder_layer.hpp`.
  - Set `layer_idx_ = layer_idx;` in the decoder layer constructor.
  - After this fix, `_infinilm` can compile; `rebuild.sh` now proceeds past the decoder layer and updates the core libraries (and should be able to update `_infinilm` when the entire build/install completes successfully).

---

### Open issues / next steps

- **1. Get the new `_infinilm` into use**
  - Ensure `rebuild.sh` completes the `_infinilm` build + install step successfully (no early termination due to missing libffi/openssl/ca‑certificates link checks).
  - Confirm via:
    ```bash
    strings /home/zenghua/repos/InfiniLM/python/infinilm/lib/_infinilm.cpython-312-x86_64-linux-gnu.so \
      | grep -E 'attn_pre_gate_l1|attn_post_gate_l1|attn_post_oproj_l1'
    ```
    If this prints the `_l1` labels, the new binary is in place.

- **2. Re‑run sanity and capture layer‑1 attention logs**
  - With the updated `_infinilm`, re‑run the prefill sanity script and inspect `debug-9146ea.log` for:
    - `minicpm_sala_attention.cpp:attn_pre_gate_l1`
    - `minicpm_sala_attention.cpp:attn_post_gate_l1`
    - `minicpm_sala_attention.cpp:attn_post_oproj_l1`
  - Compare their `l2` to HF layer‑1 (`HF_L` `l2 ≈ 73.9`).
  - This will tell us whether the **first GLA layer** is where Inf starts to diverge in norm, or whether norms remain close through layer 1 and drift later.

- **3. Use decoder `INF_L` logs to see per‑layer drift**
  - Once `_infinilm` is rebuilt, `MiniCPMSALADecoderLayer`’s per‑layer `INF_L` logs for `layer_idx_ < 3` should appear in `debug-9146ea.log`.
  - By comparing HF (`HF_L`) vs Inf (`INF_L`) for layers 0/1/2, we can see exactly where norm ratios deviate from ~1 and head toward ~0.244 at the logits.
  - That will guide targeted fixes in:
    - GLA gating / normalization (in `minicpm_sala_attention.cpp`), and/or
    - MuP residual & MLP scaling (still matching HF in formula, but potentially interacting differently with the SALA configuration).

---

### Summary

- **Plumbing**: Shared log path and HF/Inf instrumentation are in place; per‑layer HF stats and layer‑0 Inf attention stats work and confirm that **layer‑0 attention output scale is roughly aligned**.
- **Mismatch**: Final logits norm is still **Inf/HF ≈ 0.244**, so the discrepancy is accumulating across layers, likely starting at or after the first GLA layer.
- **Blocking issue**: The `_infinilm` C++ extension in use predates the layer‑1 logging changes; an earlier C++ compile error prevented a fresh install. That decode‑layer bug has been fixed so we can now rebuild and get the new diagnostics into the runtime.
- **Next milestone**: Successfully rebuild `_infinilm`, confirm the `_l1` log strings are present, rerun sanity, and use the new layer‑1 and decoder `INF_L` stats to precisely locate where Inf’s norms start drifting away from HF.

---

### Host follow-up (2026-03-14)

- Ran `examples/minicpm_sala_logits_sanity.py --mode prefill --prompt "How are you"` directly on the host using the local venv and the same base env as the documented `jiuge.py` run.
- Extra host-only prep required for the HF reference path:
  - installed `flash-linear-attention` to provide the `fla` module
  - installed `triton==3.2.0` to avoid the Triton `STAGE` autotune import failure
  - created `/home/zenghua/repos/.cursor/` because the script hardcodes `DEBUG_LOG_PATH` there
- Result on host:
  - `SANITY_ONELINE ratio=0.6215 max_diff=11.5391 mean_diff=2.5607`
  - HF top-1 token id `74`, Inf top-1 token id `23917`
- Interpretation:
  - The host environment now reproduces the alignment issue without Docker.
  - The ratio is better than the older container snapshot (`~0.244`) but still far from aligned, so the poor generation quality remains consistent with a real logits mismatch.
- Full reproducibility details for this host run were appended to `CURRENT_PROGRESS.md`.

---

### HF MiniCPM4 dense-fallback experiment (2026-03-14)

- Goal:
  - Test whether the remaining mismatch is coming from the HF `minicpm4` sparse-vs-dense code path by forcing `minicpm4` layers onto the standard dense attention implementation.
- HF model-file change:
  - Patched both cached copies of `modeling_minicpm_sala.py` so `MiniCPMSALADecoderLayer` uses `MINICPM_ATTENTION_CLASSES[config._attn_implementation]` for `mixer_type == "minicpm4"` instead of `MiniCPMInfLLMv2Attention`.
  - Backups:
    - `/root/.cache/modelscope/hub/models/OpenBMB/MiniCPM-SALA/modeling_minicpm_sala.py.bak-20260314-210428`
    - `/root/.cache/huggingface/modules/transformers_modules/MiniCPM-SALA/modeling_minicpm_sala.py.bak-20260314-210619`
- Rerun result:
  - `SANITY_ONELINE ratio=0.6215 max_diff=11.5391 mean_diff=2.5607`
  - HF top-1 token id `74`, Inf top-1 token id `23917`
  - These numbers are unchanged from the earlier host run.
- Fresh per-layer log from `debug-9146ea.log`:
  - HF decoder output `l2`:
    - layer 0: `59.49`
    - layer 1: `73.91`
    - layer 2: `87.38`
  - Inf decoder output `l2`:
    - layer 0: `35.08`
    - layer 1: `295.86`
    - layer 2: `531.38`
  - Inf layer-1 attention stats:
    - pre-gate `l2 ~= 749.58`
    - post-gate `l2 ~= 745.29`
    - post-`o_proj` `l2 ~= 1112.6`
- Interpretation:
  - For this short prefill case, forcing HF `minicpm4` to the dense fallback path does not move the mismatch at all.
  - The strongest current evidence is that the large norm drift starts in the InfiniLM implementation at or immediately after the first `lightning-attn` layer, not in the HF `minicpm4` branch.

---

### InfiniLM MiniCPM4 HF-math experiment (2026-03-14)

- Goal:
  - Make the InfiniLM `minicpm4` layer compute the same dense attention math as the HF reference path and see whether layer 0 aligns at the start of sanity.
- C++ change:
  - In `csrc/models/minicpm_sala/minicpm_sala_attention.cpp`, replaced the `minicpm4` sparse/varlen/grouped fallback branch with an explicit HF-style dense path:
    - repeat KV heads to `num_attention_heads`
    - compute per-head dense causal attention
    - keep the same sigmoid output gate and `o_proj`
- Rebuild:
  - Rebuilt and reinstalled `_infinilm` successfully using the local `xmake` toolchain.
- Rerun result:
  - `SANITY_ONELINE ratio=0.6215 max_diff=11.5391 mean_diff=2.5607`
  - HF top-1 token id `74`, Inf top-1 token id `23917`
  - These numbers are unchanged.
- Fresh layer stats after the InfiniLM-side change:
  - HF decoder output `l2`: `59.49 -> 73.91 -> 87.38`
  - Inf decoder output `l2`: `35.08 -> 295.86 -> 531.38`
  - Inf layer-0 attention:
    - pre-gate `142.87`
    - post-gate `80.43`
    - post-`o_proj` `135.39`
- Interpretation:
  - Even after making the InfiniLM `minicpm4` branch follow the HF dense attention structure, layer 0 does not move toward HF.
  - This strongly suggests the remaining mismatch is not in the `minicpm4` attention branch itself; attention should shift to other decoder-path components and especially the first `lightning-attn` layer.

---

### Temporary all-lightning experiment (2026-03-14)

- Goal:
  - Force both HF and InfiniLM to use lightning-style attention math for former `minicpm4` layers as a temporary precision-alignment probe, without changing checkpoint tensor shapes.
- Why not use `config.json` only:
  - A direct `mixer_types -> all lightning-attn` config edit failed during HF weight load because former `minicpm4` layers have incompatible checkpoint shapes for the stock `LightningAttention` module (e.g. `256 x 4096` vs `4096 x 4096`).
  - The original `mixer_types` config was restored.
- Temporary override implementation:
  - Added env flag `MINICPM_SALA_FORCE_ALL_LIGHTNING=1`.
  - HF side:
    - former `minicpm4` layers instantiate `MiniCPMAttention` under the flag
    - `MiniCPMAttention.forward()` switches to lightning-style GLA computation under the flag, while keeping original q/k/v/o_proj/o_gate weights
  - InfiniLM side:
    - `minicpm_sala_attention.cpp` routes sparse layers through `gla_attention` under the same flag
  - Sanity script:
    - `examples/minicpm_sala_logits_sanity.py` now sets `MINICPM_SALA_FORCE_ALL_LIGHTNING=1` for this experiment
- Result:
  - `SANITY_ONELINE ratio=0.4728 max_diff=12.1406 mean_diff=1.9942`
  - HF top-1 token id `59375`, Inf top-1 token id `59358`
- Fresh per-layer stats under the override:
  - HF decoder output `l2`:
    - layer 0: `385.10`
    - layer 1: `374.87`
    - layer 2: `426.87`
  - Inf decoder output `l2`:
    - layer 0: `26.23`
    - layer 1: `208.72`
    - layer 2: `403.90`
  - Inf layer-0 attention:
    - pre-gate `105.50`
    - post-gate `60.38`
    - post-`o_proj` `98.66`
  - Inf layer-1 attention:
    - pre-gate `672.74`
    - post-gate `459.67`
    - post-`o_proj` `737.03`
- Interpretation:
  - The override is definitely active on both sides, because HF logits/top-1 and HF early-layer norms changed substantially.
  - However, the former `minicpm4` layers still do not align numerically with InfiniLM under lightning-style attention.
  - This points to a mismatch in the lightning formulation itself (decay/slopes, layout, gating, norm/casting, or related details), not just in the original mixed `mixer_types` layout.

---

### Layer-0 narrowing after matched temporary semantics (2026-03-14)

- Change:
  - Updated the temporary HF override so its former `minicpm4` path uses the same grouped causal-softmax math as `InfiniCore` `gla_attention`, instead of `simple_gla` with decay.
  - Added layer-0 sub-stage logging on both sides:
    - HF: `inputs_embeds`, `input_layernorm`, `attn_pre_gate`, `attn_post_oproj`
    - Inf: embedding output, `input_layernorm`, `attn_pre_gate`, `attn_post_oproj`
- Result:
  - Layer-0 pre-gate attention still mismatches strongly:
    - HF `attn_pre_gate l2 ~= 235.11`
    - Inf `attn_pre_gate l2 ~= 105.50`
    - `Inf/HF ~= 0.4487`
  - But this is no longer the earliest divergence.
- New root-cause evidence:
  - Embedding output already differs:
    - HF `inputs_embeds l2 ~= 44.09`
    - Inf embed output `l2 ~= 25.51`
  - First decoder layer pre-norm output also differs:
    - HF layer0 `input_layernorm l2 ~= 95.88`
    - Inf layer0 `input_layernorm l2 ~= 70.94`
- Interpretation:
  - The mismatch starts before layer-0 attention.
  - Attention, gating, and `o_proj` are downstream amplifiers, but not the first source.
  - The next priority should be MiniCPM-SALA embedding behavior in InfiniLM:
    - verify `model.embed_tokens.weight` load/scaling,
    - verify runtime embedding lookup output against HF for the same token ids,
    - then re-check whether layer-0 attention comes into line automatically.

---

### Multi-layer alignment after embed fix (2026-03-14)

- Instrumentation added:
  - InfiniLM dumps decoder layer outputs (out2) for layers 0–2 to `/tmp/inf_layer_out_{0,1,2}.bin` and final hidden (after norm) to `/tmp/inf_final_hidden.bin` when `INFINI_DEBUG_ATTN_DUMP=1`.
  - HF hooks save layer outputs to `/tmp/hf_layer_out_{0,1,2}.pt` and final hidden to `/tmp/hf_final_hidden.pt`.
  - Sanity script prints per-layer and final-hidden norm_ratio and max/mean diff.
- Result (prefill "How are you", int32 input_ids workaround):
  - **Layer 0**: norm_ratio ≈ 1.0002, max_diff ≈ 0.0625 → aligned.
  - **Layer 1**: norm_ratio ≈ 3.24, max_diff ≈ 28.4 → large divergence.
  - **Layer 2**: norm_ratio ≈ 5.73 → further drift.
- Root cause for layer 1+:
  - Config: layer 0 = `minicpm4` (sparse/dense), layer 1+ = `lightning-attn`.
  - HF `LightningAttention` uses **Simple GLA** (`chunk_simple_gla` / `fused_recurrent_simple_gla`): linear/recurrent attention with decay (g_gamma), not causal softmax.
  - InfiniLM uses `op::gla_attention` for lightning layers: **causal softmax** (QK^T scale, softmax, @V). Different formulation → different scaling and dynamics.
- Next step to align after layer 0:
  - Implement Simple GLA (chunk or fused_recurrent) in InfiniCore and route lightning layers through it, matching HF’s `attn_fn` (decay, scale=1/sqrt(d), layout).
