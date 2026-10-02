# MoE Expert Grading — llama.cpp fork

llama.cpp fork for **per-tier MoE expert quantization**. Each layer's experts
are split into hot / warm / cold tiers, and each tier is quantized to its own
bit width. Based on upstream `0c1e570`.

> **This fork was written by an AI coding agent**, working from a written
> design and from benchmark measurements taken while developing it. The
> numbers in this repo are real and reproducible with the scripts in
> `llama.cpp-fork/`, but no human outside the project has reviewed the code.
> Read the patch before relying on it.

```sh
git clone --depth 1 --branch 0c1e570 https://github.com/ggml-org/llama.cpp
cd llama.cpp && patch -p1 < graded-moe-0c1e570.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j
```

## Release binaries

**`llama-fork-vulkan-graded-win-x64.zip`** (20 MB) — Windows x64, CPU + Vulkan,
built from this fork with the repack guard. Verified on an AMD Radeon
RX 5700 XT (8 GB).

```sh
unzip llama-fork-vulkan-graded-win-x64.zip
cd llama-fork-vulkan-graded
llama-cli.exe --list-devices
```

**Scope of verification:** Qwen3.6-35B-A3B only (256 experts per layer, top-8,
40 layers). Other architectures are untested. Speculative decoding (draft
models, MTP) has not been verified with tiered expert tensors.

## Tiered expert tensors

Instead of one `ffn_gate_exps.weight` per layer, the model stores one tensor per
tier — `ffn_gate_exps_hot`, `_warm`, `_cold` — independently for gate / up /
down, each in its own quantization type. One extra F32 tensor
`moe_expert_groups.N` per layer records the tier boundaries.

Tiers come from ranking each layer's experts by activation energy measured with
an importance matrix: top 20% hot (51 experts), next 30% warm (77), rest cold
(128).

At graph build time `build_moe_mm_id_grp` dispatches every `mul_mat_id`
through a lookup table over tier ids plus a mask of the tokens routed inside
each tier, so all tiers are served in a single pass while the router keeps
seeing the full expert space. Only 8 of 256 experts fire per token, which is why
the cold tier can sit three bits below the hot tier at almost no perplexity cost.

Non-expert tensors (router, norms, attention, shared experts, output) keep their
normal types.

## Also fixes a CPU repack bug

`get_optimal_repack_type()` returns a repack trait for Q4_K/Q5_K without
checking dimensionality, so 3D expert tensors with `ne[2] % 8 != 0` (51 / 77)
take the 8×8 interleaved path and compute wrong results — PPL 3.7313 instead of
1.5085 on the same file. Upstream merged models are unaffected (`ne[2] = 256`).
The guard returns `nullptr` for 3D tensors and costs no throughput (pp512
110.88 vs 110.15 t/s, tg64 17.42 vs 16.85 t/s, both inside error bars).

## Files

| path | what |
|---|---|
| `llama.cpp-fork/patches/graded-moe-0c1e570.patch` | the patch (md5 `7f2cd69518f4ae8c7e4ab8fa20259a57`) |
| `llama.cpp-fork/quant_recipe.py` | one-command quantization, dry-run by default |
| `llama.cpp-fork/split_moe_experts.py` | expert tiering + `moe_expert_groups` |
| `llama.cpp-fork/fix_imatrix_names.py` | tier imatrix aliases |
| `llama.cpp-fork/README-TECH.md` | full write-up: pitfalls, storage, reproduction |

MIT, same as upstream.