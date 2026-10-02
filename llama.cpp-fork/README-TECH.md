# Graded-MoE fork: per-tier expert quantization for Qwen3.6-35B-A3B

Splits every MoE expert tensor into three tiers (hot/warm/cold) so each tier can
be quantized to a different type, then runs them through a llama.cpp fork that
reassembles them at inference time.

## Headline results (same corpus, same binary, -t 20)

| model | size | PPL |
|---|---|---|
| F16 | 67 GB | 1.5053 |
| **graded v8 (this repo)** | **17.26 GiB** | **1.5030** |
| graded v9 | 16.61 GiB | 1.5085 |
| graded v5b (uniform Q3_K, no gradient) | 15.82 GiB | 1.5180 |
| APEX I-Compact (upstream, 16.1 GiB) | 16.1 GiB | 1.5125 |
| upstream Q4_K_S / Q3_K_S | 20 GB / 15 GB | 1.5167 / 1.5404 |

v8 recipe (per tier: gate / up / down):

| tier | experts | gate | up | down |
|---|---|---|---|---|
| hot | 51 | q4_K | q4_K | q5_K |
| warm | 77 | q3_K | q3_K | q4_K |
| cold | 128 | q3_K | q3_K | q3_K |

Non-expert tensors stay high precision: router + norms F32, shared experts +
output Q8_0, attention + token_embd Q6_K.

## 1. Build the fork

```sh
# upstream source (any recent master works)
curl -L -o lc.tgz https://ghfast.top/https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz
tar -xzf lc.tgz -C /root

cd /root/llama.cpp-master
patch -p1 < fork_0c1e570.patch      # 10 files; see md5 below

cmake -G Ninja -B build -S . \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_VULKAN=OFF -DGGML_CUDA=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
ninja -C build llama-perplexity
```

The patch adds: `moe_expert_groups.<il>` metadata, per-tier tensor enums, the
grouped `build_moe_mm_id_grp` graph path (LUT + mask), one explicit template
instantiation for `get_arr<std::string, int>`, and the repack guard of section 4.

## 2. Quantization pipeline

```sh
# 2.1 BF16 safetensors -> F16 gguf. ALWAYS pass --outtype f16.
python3 llama.cpp-master/convert_hf_to_gguf.py /root/Qwen3.6-35B-A3B \
    --outfile /root/qwen35moe_f16.gguf --outtype f16

# 2.2 optional importance matrix (10 chunks x 512 tokens in our runs)
llama-b9561/llama-imatrix -m /root/qwen35moe_f16.gguf -f calib.txt

# 2.3 split experts into tiers + write moe_expert_groups metadata
export TMPDIR=/root/tmp && mkdir -p /root/tmp
python3 split_moe_experts.py -i /root/qwen35moe_f16.gguf -o /root/split_f16.gguf \
    -t profile/tiers_per_layer.json --hot 0.20 --warm 0.30
# must print: verified 360 tiered tensors, 768 expert slabs compared,
#             all 633 passthrough tensors shape+byte checked

# 2.4 tier aliases in the imatrix (sorted! see 4.2)
python3 fix_imatrix_names.py ...        # -> agent_split.imatrix.gguf

# 2.5 rules + quantize + perplexity (one command)
python3 quant_recipe.py --recipe v8 --target-mib 17673 --run --ppl
python3 quant_recipe.py --list          # 6 built-in recipes
python3 quant_recipe.py --recipe v8 --set cold.gate=q4_K --dry-run   # tweak
```

`quant_recipe.py` defaults to a dry-run: it prints the per-tier type table,
the predicted size (calibrated against two measured points, ±1 MiB), the 367
rule lines and the type histogram, then refuses to run if the prediction misses
`--target-mib` by more than `--tol`. Artifacts are derived from the recipe name:

```
/mnt/workspace/moe/ttypes_<recipe>.txt    367 rules = 7 keep + 40 layers x 3 proj x 3 tiers
/root/graded-<recipe>.gguf                output model
/root/quant_<recipe>.log / .ppl.log       quantize log / perplexity log
```

## 3. Perplexity

```sh
build/bin/llama-perplexity -m /root/graded-v8.gguf \
    -f /mnt/workspace/ppl_small.txt -t 20
# Final estimate: PPL = 1.5030 +/- 0.02916
```

Judge against the F16 ceiling (1.5053), not against zero. ±0.029 is the
run-to-run band, so two recipes closer than that are tied.

## 4. Pitfalls (all of them cost real time)

### 4.1 Never feed BF16 to the split script
`gguf-py`'s reader returns `uint8` for BF16. The split script must not do
`.view(np.float16)` on it: that reinterprets bits (BF16 = f32's top 16) and
silently destroys every weight. Symptom: PPL in the e35 range. Always convert
with `--outtype f16` first and assert the reader dtype is `float16`.

### 4.2 Tier imatrix rows must be in ascending expert id order
`split_moe_experts.py` writes expert slabs with `np.nonzero(grp == g)`, which is
ascending. `fix_imatrix_names.py` must `sorted(...)` the same slice, otherwise
every expert is quantized with another expert's importance matrix. Symptom: none
-- size checks and `imatrix size` warnings stay clean, only PPL suffers.

### 4.3 ggml repack is not compatible with 3D expert tensors
`get_optimal_repack_type()` ships no Q3_K trait (so Q3_K takes the plain path)
but does return a trait for Q4_K/Q5_K, which routes `MUL_MAT_ID` into the
repacked `8x8` interleaved layout. On a grouped model with `ne[2] = 51/77` that
layout disagreed with the expert slicing:

```
same model file:  3.7313 with repack,  1.5085 with repack disabled
upstream ne[2]=256 (256 % 8 == 0): unaffected, 1.5167
```

The patch guards it (`get_optimal_repack_type` returns nullptr for 3D tensors).
Alternative: configure with `-DGGML_CPU_REPACK=OFF`.
Bench says the guard costs nothing (pp512 110.15 vs 110.88 t/s, tg64 16.85 vs
17.42 t/s, both inside error bars) because MoE activates ~3% of experts, which
is not the access pattern the interleaved layout optimizes for.

### 4.4 llama-b9561 rejects blk.40 on the pure-ftype path
`Bad layer 40 for tensor blk.40.ffn_down_exps.weight. Must be in [0, 40)` --
the model has 41 blocks (MTP) and b9561 clamps to `block_count - 1`. The
per-tensor rule path does not run that check, so pin it:
`blk\.40\..*=q8_0`. blk.40 is the MTP block, it is logged as an unused tensor
and never affects perplexity.

### 4.5 ttypes rules are per-projection
One rule per layer per tier per projection (360 + 7 keep). A `.*` wildcard
across projections (`blk\.1\.ffn_.*_exps_hot=q4_K`) silently disables any
down-vs-gate/up gradient -- it looks valid and quantizes fine.

### 4.6 IQ series needs a much bigger calibration set
IQ4_XS / IQ3_XXS are built around the importance matrix. With only 5,120
calibration tokens (10 chunks x 512) from a single domain they collapsed to
PPL 2.10 while plain Q3_K on the identical split stayed at 1.518. Use K-series
types until the imatrix is re-run on ~50k tokens across several domains.

## 5. Files

| file | md5 | note |
|---|---|---|
| `fork_0c1e570.patch` (38,945 B) | `7f2cd69518f4ae8c7e4ab8fa20259a57` | 10 files, includes the repack guard |
| `quant_recipe.py` (17,330 B) | `331ab6eb263c4b0637801b6693ce1060` | dry-run first, per-recipe artifacts |
| `split_moe_experts.py` | -- | tier slicing + `moe_expert_groups` KV + byte verification |
| `fix_imatrix_names.py` | -- | tier imatrix aliases (ascending order) |
| `parse_imatrix_profile.py` | -- | `expert_order_by_energy` from the imatrix |
| `gen_quant_recipe.py` / `parse_imatrix_profile.py` | -- | earlier recipe generator |
| `upstream_baseline.sh` (2,730 B) | `ec7762c38f6dfe8c46a9e5bc9158648b` | upstream Q4_K_S vs Q3_K_S control |
| `verify_norepack.sh` (2,764 B) | `16c5d142469cceced31a51eaa045b6a5` | proves the repack verdict end to end |

## 6. Storage

`/root` is a container disk and is wiped whenever the container is replaced.
Keep on the persistent volume (`/mnt/workspace/...`): the patch, the scripts,
the imatrix files, `ttypes_*.txt`, and every finished `graded-*.gguf`. The 71 GB
F16 intermediates can be rebuilt from `convert` + `split` in about 10 minutes.
