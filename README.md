# llama.cpp — Graded-MoE fork

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at
`0c1e570` (webgpu: fix SSM_SCAN binding aliasing, #29750) carrying two
independent changes for **per-tier expert quantization** of Mixture-of-Experts
models, verified on Qwen3.6-35B-A3B (256 experts / layer, top-8, 40 layers):

1. **Per-tier expert tensors** — MoE expert weights can be split into
   `hot` / `warm` / `cold` tiers, each quantized to a different type, while the
   router keeps routing over the full 256-expert space.
2. **CPU repack guard** — a bug fix: the CPU weight-repack path produces
   corrupted results on 3D expert tensors whose expert count is not a multiple
   of 8. (See [repack bug](#cpu-repack-bug-on-3d-tensors).)

Both changes are in a single 10-file patch, 38,945 bytes, md5
`7f2cd69518f4ae8c7e4ab8fa20259a57`, which applies cleanly to the upstream
commit above with `patch -p1`.

## Results

Qwen3.6-35B-A3B, same 14-chunk corpus, `-t 20`, single machine:

| model | size | PPL |
|---|---:|---:|
| F16 | 67 GB | 1.5053 |
| **graded v8** (Q4_K / Q3_K / Q2_K-K-mixed tiers) | **17.26 GiB** | **1.5030** |
| graded v10 | 15.59 GiB | 1.5096 |
| APEX I-Compact (upstream) | 16.10 GiB | 1.5125 |
| upstream Q4_K_S | 20 GB | 1.5167 |
| upstream Q3_K_S | 15 GB | 1.5404 |

Run-to-run variance is ±0.029. The graded variants are within ~0.004 of the
F16 ceiling; v10 is 0.51 GiB smaller than APEX with a better PPL.

Threading, same model, this fork with the guard enabled:

| test | t/s |
|---|---:|
| pp512 | 110.88 ± 3.65 |
| tg64 | 17.42 ± 0.47 |

## How per-tier tensors work

Per layer, the 256 experts are ranked by activation energy (from an imatrix)
and split into tiers: 20% hot, 30% warm, 50% cold. The model stores, per
layer and per projection (gate / up / down), one tensor per tier —
e.g. `blk.N.ffn_gate_exps_hot`, `blk.N.ffn_gate_exps_warm`,
`blk.N.ffn_gate_exps_cold` — each in its own quantization type, plus one
F32 tensor `moe_expert_groups.N` describing the tier→expert-id mapping.

At load time the reader registers the tier tensors and the group metadata;
at graph build time each `mul_mat_id` is dispatched through
`build_moe_mm_id_grp`, which uses a small lookup table over tier ids plus a
mask of tokens whose expert ids fall inside the tier, so all three tiers are
served by one graph pass over the same 256-wide router output. Non-expert
tensors (router, norms, attention, shared experts, output) keep their normal
types.

## CPU repack bug on 3D tensors

`ggml_repack_get_optimal_repack_type()` in `ggml/src/ggml-cpu/repack.cpp`
has a trait for `Q4_K` / `Q5_K` (but not `Q3_K`) and does not check the
tensor's dimensionality. A 3D expert tensor with `ne[3] == 1`, `ne[2] > 1`
and `ne[2] % 8 != 0` (our tier sizes are 51 / 77 / 128) is therefore routed
into the 8×8 interleaved repack layout, which does not match how
`forward_mul_mat_id` slices expert data:

| same model file | PPL |
|---|---:|
| repack enabled (trait matches) | 3.7313 |
| repack disabled (`GGML_CPU_REPACK=OFF`) | 1.5085 |
| upstream merged model, `ne[2] = 256` (256 % 8 == 0) | 1.5167 — unaffected |

The fix is a guard in `get_optimal_repack_type()`: 3D tensors return
`nullptr` and fall back to the plain path. Bench on the 17 GiB model shows
no measurable cost (pp512 110.15 vs 110.88 t/s, tg64 16.85 vs 17.42 t/s,
both inside error bars) — MoE activates ~3% of experts per token, which is
not the access pattern the interleaved layout optimizes for.

This is an upstream-eligible fix; the guard is ~3 lines.

## Building

```sh
git clone --depth 1 --branch 0c1e570 https://github.com/ggml-org/llama.cpp
cd llama.cpp
patch -p1 < path/to/fork.patch     # or: git apply fork.patch

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

GPU backends are orthogonal and unaffected. The guard is compiled in
regardless of `-DGGML_CPU_REPACK`; with `GGML_CPU_REPACK=OFF` it is a no-op.

## Verification checklist (reproducible)

| step | command / artifact |
|---|---|
| patch applies | `git apply --check fork.patch` against `0c1e570` |
| tier tensors correct | `split_moe_experts.py` byte-compares all 768 expert slabs + 633 passthrough tensors against the source |
| repack verdict | `verify_norepack.sh`: two builds (`GGML_CPU_REPACK` ON/OFF), one model, one corpus → 3.73 vs 1.51 |
| PPL | `llama-perplexity -m graded-v8.gguf -f corpus -t 20` → 1.5030 |

## Files changed

| file | change |
|---|---|
| `ggml/src/ggml-cpu/repack.cpp` | 3D guard in `get_optimal_repack_type()` |
| `ggml/include/llama/ggml-llama.h` | tier tensor enum + group metadata key |
| `src/llama-arch.cpp` | tier tensor registration for MoE archs |
| `src/llama-graph.cpp` | `build_moe_mm_id_grp` (LUT + mask), dispatch |
| `src/llama-model.cpp` | reader: tier tensors + `moe_expert_groups` |
| `src/llama-mmap.cpp` (+4 more) | support / instantiation (`get_arr<std::string,int>`) |

Full list and per-file diffs in `fork_0c1e570.patch`.

## Status

- [x] Local (Windows / MSVC) build with guard: `--version` OK, all binaries
- [x] Full pipeline on cloud Linux build: v8 1.5030, v10 1.5096
- [ ] Upstream PR for the repack guard (pending)
- [ ] Windows binary release

MIT, same as upstream.
