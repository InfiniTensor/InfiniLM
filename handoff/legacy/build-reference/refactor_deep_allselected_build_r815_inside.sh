#!/bin/bash
set -euo pipefail
BASE=/refactor/candidates/native-engine-allselected-r815
LM=/refactor/candidates/native-engine-memory-r23/InfiniLM
OLD=/refactor/src/InfiniLM
SDK=/refactor/install/hygon-v4-indexer-ops-r1
export LD_LIBRARY_PATH="$BASE:/refactor/build:$SDK/lib:$OLD/python:$OLD/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$OLD/python:/old/offline-deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
python - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('/refactor');q=json.loads((r/'results/deepseek-policy-r440-910373.353-summary.json').read_text());assert q['completed']and q['owned_BLAS_policy_verified']and len(q['engine_cases'])==2 and len(q['chat_cases'])==6
assert hashlib.sha256((r/'candidates/native-engine-selection-r356/libmodern_native_engine_selection_r356.so').read_bytes()).hexdigest()==q['engine_library_sha256']
PY
# Original CPU selection equivalence and new data movement audit gate the model.
python - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('/refactor');q=json.loads((r/'results/deep-gather-r381-audit.json').read_text());assert q['passed']and q['attention_cases']==68 and q['fullBF16domain_layout_cases']==32
assert hashlib.sha256((r/'candidates/deepseek-latent-gather-r374/gather.o').read_bytes()).hexdigest()==q['gather_object_sha256']
assert json.loads((r/'results/deep-selection-component-r356-summary.json').read_text())['passed']
PY
python - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('/refactor');q=json.loads((r/'incoming/deep-cache-batch-r784-originals.json').read_text())
for n,h in q.items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h
q=json.loads((r/'results/native-engine-fused-latent-r686-910373.404.json').read_text());assert q['completed']and q['own_long_repeat_exact']and q['engine_worker_shutdown_completed']
PY
python - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('/refactor');q=json.loads((r/'results/deep-c4-allselected-r814-910373.435-audit.json').read_text());assert q['completed']and q['cases']==432 and len(q['owners'])==8 and q['all8_reduced_scores_and_selected_indices_identical']
launch=json.loads((r/'deep-c4-allselected-r813-r1-launch.json').read_text())
for n,h in launch['source_sha256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h
q=json.loads((r/'results/native-engine-cache-batch-r787-910373.428.json').read_text());assert q['completed']and q['own_long_repeat_exact']and q['engine_worker_shutdown_completed']and all(c['frozen686_full_IDs_rawfirst4_exact']for c in q['cases'])
PY
ABI=$(python -c 'import torch;print(int(torch.compiled_with_cxx11_abi()))')
mkdir -p "$BASE"
test ! -e "$BASE/libmodern_native_engine_allselected_r815.so"
python - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('/refactor');q=json.loads((r/'results/deep-latent-fused-r684-r681-summary.json').read_text());assert q['passed']and q['cases']==98 and q['invalid_recovery_checks']==240
assert q['by_shape']['csa8']['min_speedup']>1 and q['by_shape']['csa']['min_speedup']>1
assert hashlib.sha256((r/'candidates/deepseek-latent-fused-r681/fused.o').read_bytes()).hexdigest()==q['kernel_object_sha256']
for n,h in q['source_sha256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h
PY
# Reuse the already built, unchanged Sinkhorn object from633.
python - <<'PY'
import json,pathlib
r=pathlib.Path('/refactor');q=json.loads((r/'results/mhc-post-register-r168-summary.json').read_text());assert q['all_cases_collected']and q['selected_variant']==1 and q['qualified_byte_exact_variants']==[1]
assert json.loads((r/'results/deepseek-b8-r114-910373.280-summary.json').read_text())['completed']
q=json.loads((r/'results/deepseek-prefill-scatter-r194-summary.json').read_text());assert q['passed'] and q['numeric_checks']==192
q=json.loads((r/'results/deepseek-latent-packed-r227-summary.json').read_text());assert q['passed'] and len(q['cases'])==68 and all(q['library_owner_shutdown_completed'])
PY
g++ -std=c++17 -O2 -shared -fPIC -pthread -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
 -I"$LM/csrc" -I"$OLD/csrc/models/deepseek_v4" -I/refactor/incoming -I"$SDK/include" -I"$SDK/include/infiniccl" -I"$OLD/third_party/json/include" -I"$OLD/third_party/spdlog/include" -I/opt/dtk/cuda/cuda/include \
 "$LM/csrc/native_sparse_model_r22.cpp" "$LM/csrc/config/config_factory.cpp" "$LM/csrc/config/model_config.cpp" "$LM/csrc/config/quant_config.cpp" \
 "$LM/csrc/models/models_registry.cpp" "$LM/csrc/models/model_factory.cpp" "$LM/csrc/engine/infer_engine.cpp" "$LM/csrc/engine/rank_worker.cpp" \
 /refactor/incoming/modern_native_engine_probe_r22.cpp \
 /refactor/incoming/deepseek_v4_routed_text_model_r20.cpp /refactor/incoming/deepseek_v4_native_routed_decoder_r20.cpp /refactor/incoming/deepseek_v4_ffn_routed_block_r20.cpp /refactor/incoming/deepseek_v4_hyper_connection_r177.cpp /refactor/incoming/deepseek_v4_moe_prefill_scatter_r194.cpp /refactor/build/deepseek_routed_i8_kernels_r20.o /refactor/incoming/deepseek_v4_int8_mlp_bank_r20.cpp /refactor/incoming/deepseek_v4_swa_attention.cpp /refactor/incoming/deepseek_v4_hca_attention_batch_r784.cpp /refactor/incoming/deepseek_v4_csa_attention_batch_r784.cpp /refactor/incoming/deepseek_v4_c4_indexer_allselected_r813.cpp /refactor/incoming/deepseek_v4_compressor_fp32_c128_r19.cpp /refactor/incoming/deepseek_v4_paged_latent_cache_public.cpp /refactor/incoming/deepseek_v4_attention_prepare.cpp /refactor/incoming/deepseek_v4_attention_output.cpp /refactor/incoming/deepseek_v4_latent_model_r685.cpp /refactor/incoming/deep_latent_reference_r681.cpp \
 /refactor/candidates/native-engine-fp32-reduce-r633/sinkhorn.o /refactor/candidates/deepseek-latent-gather-r374/gather.o /refactor/candidates/deepseek-latent-fused-r681/fused.o /refactor/candidates/mhc-post-register-r168/post.o /refactor/candidates/deepseek-prefill-scatter-r194/scatter.o \
 /refactor/incoming/deep_cache_batch_counts_r784.cpp -L/refactor/build -lmodern_glm_routed_pipeline_r19 \
 -L"$OLD/python" -l:_infinilm.cpython-310-x86_64-linux-gnu.so -L"$OLD/python/infinilm/_lib" -linfinilm_runtime \
 -L"$SDK/lib" -linfiniops -linfiniccl -linfinirt -L/opt/dtk/cuda/cuda/lib64 -lcudart \
 -Wl,-Bsymbolic -Wl,--no-undefined -o "$BASE/libmodern_native_engine_allselected_r815.so"
python - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('/refactor');base=r/'candidates/native-engine-allselected-r815';lib=base/'libmodern_native_engine_allselected_r815.so';path=r/'results/deepseek-allselected-r815-build.json';assert not path.exists()
names=['incoming/deep_cache_batch_r784.hpp','incoming/deep_cache_batch_counts_r784.cpp','incoming/deepseek_v4_csa_attention_batch_r784.cpp','incoming/deepseek_v4_hca_attention_batch_r784.cpp','incoming/deepseek_v4_c4_indexer_allselected_r813.cpp','incoming/deep-cache-batch-r784-originals.json','incoming/deepseek_v4_latent_model_r685.cpp','incoming/deep_latent_reference_r681.cpp','incoming/deep_latent_fused_r681.cu','incoming/deepseek_v4_attention_output.cpp','incoming/deepseek_v4_moe_prefill_scatter_r194.cpp','incoming/deepseek_v4_hyper_connection_r177.cpp','incoming/refactor_deep_allselected_build_r815_inside.sh','candidates/deepseek-latent-fused-r681/fused.o','candidates/deepseek-latent-gather-r374/gather.o','candidates/native-engine-fp32-reduce-r633/sinkhorn.o','candidates/mhc-post-register-r168/post.o','candidates/deepseek-prefill-scatter-r194/scatter.o','results/deep-latent-fused-r684-r681-summary.json']
record={'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'inputs_sha256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest()for n in names},'sdk_sha256':hashlib.sha256((r/'install/hygon-v4-indexer-ops-r1/lib/libinfiniops.so').read_bytes()).hexdigest(),'fused_min_tokens':8,'fused_max_tokens':64,'branch':'6b9cfa6b00cc7ca7431bc18e6c97936c4276e83e','scope':'815overqualified784/787: opt-in813canonicalallselected M8..64single-token decode whenevervisiblecompressedkeys<=topk. Skipqprojection/features/headscore/TPscore/Topk, keepkeycompressor/cachecommit forfuturepruning. 432actual7weight/TP8 componentcases preserveallstate/cache/selectionset andfullfallbackscores/indices afterboundary; attentionreductionorder intentionallychanges. FullmodelknownQA/own512repeats/raw/all8/owner/shutdown required. Parent784coalesces historical C4feature/cache andCSA/HCAcompressor gathers acrossrequests. Samecachecodec, samerequestorder andviews; fallsbackabove existingboundedcounts. Keepallqueryprojection/score/topk/TPorder/latent681 mathandSDK8865 unchanged. Requirefull686ID/raw compatibility/all8actualbatchcounters/QA/shutdown. Parent685onlylatentattention split64kernel681 overqualified482/635nativeBF16attentionAllReduce, MoEalreadyF32 unchanged. 8..64tokens+metadata<=2048 uses GPUvalidate andF32partial/stablemerge->BF16; scalarandprefillfallbackold374. Sameordinary8owner/cache16/SDK8865/271. IntentionalFP32reductionorderchange, 98component/240invalid/F64samplegate beforefullmodel. No634FP32attentionreduce orTreepolicy.'}
assert record['sdk_sha256']=='8865b8d8e67832884911199d1a667856aadb2b438f5287d17fbaff1f8a3b85db';path.write_text(json.dumps(record,indent=2));print('DEEP_ALLSELECTED815_BUILD',record['library_sha256'])
PY
