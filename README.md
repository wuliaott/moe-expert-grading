# Qwen3.6-35B-A3B — MoE Expert Grading

Per-expert graded quantization for Mixture-of-Experts models: the experts of
every layer are ranked by measured activation energy and split into three
tiers, and each tier is quantized to a different bit width. Built and verified
against a patched llama.cpp (`moe-expert-grading/llama.cpp-fork`, bundled below
as a patch).

> [!WARNING]
> **These GGUF files do not load in stock llama.cpp.** They carry extra
> `moe_expert_groups` metadata and per-tier tensors (`*_hot` / `*_warm` /
> `*_cold`). Apply `llama.cpp-fork/patches/graded-moe-0c1e570.patch` (or build
> the fork) before loading, or you get `failed to load model`.

## Which one should I download?

| | file | size | PPL (lower is better) | pick this when |
|---|---|---:|---:|---|
| **Balanced (v8)** | `Qwen3.6-35B-A3B-MoEGraded-v8.gguf` | **17.26 GiB** | **1.5030** | You want the best quality your disk can hold. Hot tier gets Q4_K gate/up plus Q5_K down, the warm tier keeps a gate/up/down gradient, and the budget stays inside 17 GB. |
| **Compact (v10)** | `Qwen3.6-35B-A3B-MoEGraded-v10.gguf` | **15.59 GiB** | **1.5096** | You want the smallest model that still matches the competition. The 128 cold experts drop gate/up to Q2_K while their down projections hold Q3_K. It is **0.51 GiB smaller than APEX I-Compact with better perplexity** (1.5096 vs 1.5125). |

Both sit within 0.007 perplexity of each other and within 0.004 of the F16
reference (1.5053). Run-to-run variance on this benchmark is ±0.029, so the two
are effectively tied on quality: v8 buys headroom, v10 buys disk.

## Perplexity results

Same corpus, 14 chunks × 512 tokens, `-t 20`, single machine:

| model | size | PPL |
|---|---:|---:|
| F16 reference | 67 GB | 1.5053 |
| **MoE Expert Grading — Balanced (v8)** | **17.26 GiB** | **1.5030** |
| MoE Expert Grading — Compact (v10) | **15.59 GiB** | **1.5096** |
| APEX I-Compact (upstream) | 16.10 GiB | 1.5125 |
| uniform Q3_K (no grading) | 15.82 GiB | 1.5180 |
| upstream Q4_K_S | 20 GB | 1.5167 |
| upstream Q3_K_S | 15 GB | 1.5404 |

## How it works

1. **Rank the experts.** An importance matrix collected over 5,120 calibration
   tokens gives every expert an activation energy. Per layer the 256 experts are
   sorted by that energy.
2. **Tier them.** Top 20% become `hot` (51 experts), the next 30% `warm` (77),
   the rest `cold` (128).
3. **Quantize per tier.** Each expert slab keeps its own tensor type, so a layer
   ends up as `ffn_gate_exps_hot` (Q4_K), `ffn_gate_exps_warm` (Q3_K),
   `ffn_gate_exps_cold` (Q2_K), and so on — independently for gate/up/down.
4. **Keep the rest precise.** Router and norms stay F32, attention and the token
   embedding Q6_K, shared experts and the output head Q8_0.
5. **Reassemble at load time.** The patched loader registers the tier tensors
   and the `moe_expert_groups` metadata; the graph builder dispatches each
   `mul_mat_id` through `build_moe_mm_id_grp`, using a lookup table over tier
   ids plus a mask for tokens whose expert ids fall inside the tier.

Only 8 of 256 experts are active per token, so a wrong tier costs nothing for
the vast majority of routed tokens — that is what lets the cold tier run three
bits below the hot tier while perplexity stays flat.

## Usage

```sh
git clone --depth 1 --branch 0c1e570 https://github.com/ggml-org/llama.cpp
cd llama.cpp
patch -p1 < graded-moe-0c1e570.patch

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

build/bin/llama-cli -m Qwen3.6-35B-A3B-MoEGraded-v8.gguf \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5 \
    -p "Write a haiku about quantization."
```

Sampling settings follow the official Qwen3.6 recommendation. Greedy decoding
triggers the model's known repetition loop.

## Repository layout

| path | what |
|---|---|
| `llama.cpp-fork/patches/graded-moe-0c1e570.patch` | the 10-file patch (md5 `7f2cd69518f4ae8c7e4ab8fa20259a57`) |
| `llama.cpp-fork/quant_recipe.py` | one-command quantization from F16, dry-run by default |
| `llama.cpp-fork/split_moe_experts.py` | expert tiering + `moe_expert_groups` metadata, byte-verified |
| `llama.cpp-fork/fix_imatrix_names.py` | tier imatrix aliases (ascending expert id order — required) |
| `llama.cpp-fork/verify_norepack.sh` | proves the CPU repack verdict end to end |
| `llama.cpp-fork/upstream_baseline.sh` | upstream Q4_K_S vs Q3_K_S control run |

## Reproducing

```sh
python3 quant_recipe.py --list
python3 quant_recipe.py --recipe v8 --target-mib 17673 --run --ppl
```

`--target-mib` makes the script assert the predicted size against the size
actually produced before it starts, so a silent recipe change cannot slip
through.