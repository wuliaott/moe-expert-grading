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
patch -p1 < graded-moe-g1.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j
```

## Release binaries

**`llama-fork-vulkan-graded-g1-win-x64.zip`** (21 MB) — Windows x64, CPU + Vulkan,
built from this fork with the repack guard **and the G1 update below**. Verified
on an AMD Radeon RX 5700 XT (8 GB). The pre-G1
`llama-fork-vulkan-graded-win-x64.zip` (v1.0) is attached to the
`v1.0-graded-moe` release.

```sh
unzip llama-fork-vulkan-graded-g1-win-x64.zip
cd llama-fork-vulkan-graded
llama-server.exe -m Qwen3.6-35B-A3B-MoEGraded-Compact.gguf --spec-type draft-mtp -ngl 99
```

**Scope of verification:** Qwen3.6-35B-A3B only (256 experts per layer, top-8,
40 layers). Other architectures are untested. Speculative decoding (MTP,
`--spec-type draft-mtp`) **is verified** with tiered expert tensors — see below.

## G1 update (2026-10): one op per MoE projection, MTP support, 2× generation

G1 replaces the three per-tier `mul_mat_id` calls with a single fused
`GGML_OP_MUL_MAT_ID_GRP`: the router's per-tier lookup tables and 0/1 ownership
masks go in, and each (slot, token) pair is evaluated exactly once against the
owning tier — **8 of 256 experts per token, computed once** — instead of
computing all 8 slots three times and masking two thirds of the work away.

Measured on the RX 5700 XT (`-t 20`, `-p 512 -n 128`):

| configuration | prompt (t/s) | generation (t/s) |
|---|---|---|
| all layers on GPU (`-ngl 99`) | 118.7 | 11.1 |
| `--n-cpu-moe 36` | 217.8 | 10.2 |
| CPU only (`-ngl 0`) | 143.0 | 6.6 |
| **all layers on GPU + `--spec-type draft-mtp`** | — | **14.6 server-measured, ~21 with tuned draft settings** |

Quality is unchanged: perplexity of the graded model equals its pre-G1 value,
and the CPU and GPU paths agree bit-for-bit on the same corpus. For reference,
the same-size standard-quantization baseline (APEX I-Compact Q4_K_M) measures
pp 76.5–113 t/s and tg 12.75 t/s on identical hardware and flags.

### Enable MTP (recommended)

MTP speculative decoding uses the model's own nextn block — no sidecar file, no
network access:

```sh
llama-server -m Qwen3.6-35B-A3B-MoEGraded-Compact.gguf --spec-type draft-mtp -ngl 99
```

The type name is `draft-mtp` (plain `mtp` is rejected). Measured draft
acceptance rate is ~70 % (≈ 3.1 accepted tokens per draft round). **This is the
recommended way to run the model** — with MTP on, generation matches a normal
(ungraded, standard-quant) Qwen3.6 build, which closes the only remaining gap
against the same-size baseline.

### What G1 fixes beyond the speedup

- **MTP speculative decoding now works.** The v1.0 build was missing the
  standard (non-masked) `mul_mat_vec_id` Vulkan pipelines, so any model path
  that runs a plain `mul_mat_id` on the GPU crashed with
  `GGML_ASSERT(dmmv != nullptr)` — this included the MTP draft context and
  normal (ungraded) qwen3.5moe models. All 41 pipeline variants are registered
  again; MTP and ungraded models load and run.
- **CPU MoE path rewritten.** The CPU fallback now filters the routing tables
  by the ownership masks and evaluates each (slot, token) pair exactly once
  (8 slots instead of 24), shares one quantized copy of the activations across
  tiers, and writes the shared output directly. With CPU-offload configurations
  the CPU side is no longer the bottleneck (pp 210→218 t/s, tg 5.3→10.2 t/s
  vs the first G1 build).
- **Legacy fallback guard.** The old per-tier fallback path (known to compute
  incorrect results on some builds) can no longer be taken silently; a graded
  model with missing groups/luts/masks now aborts with a clear message.
- Two experimental tuning knobs are included but off by default:
  `GGML_VK_SPLITK_MIN`, `GGML_VK_RM_STDQ` / `GGML_VK_RM_KQ`.

## Tiered expert tensors

Instead of one expert tensor per layer, the model carries three — one per tier,
named `ffn_gate_exps_hot`, `ffn_gate_exps_warm` and `ffn_gate_exps_cold` for the
gate projection, and the same way for up and down. Each is quantized to its own
type. One extra F32 tensor per layer, `moe_expert_groups.N`, records the tier
boundaries.

Tiers come from ranking each layer's experts by activation energy measured with
an importance matrix: top 20% hot (51 experts), next 30% warm (77), rest cold
(128).

At graph build time `build_moe_mm_id_grp` dispatches one fused
`mul_mat_id_grp` per projection: the router keeps seeing the full expert space,
each tier serves exactly the slots it owns, and the MTP block (layer 40, kept
ungraded at Q8_0) runs through the standard path. Only 8 of 256 experts fire
per token, which is why the cold tier can sit three bits below the hot tier at
almost no perplexity cost.

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
| `llama.cpp-fork/patches/graded-moe-0c1e570.patch` | the v1.0 patch (md5 `7f2cd69518f4ae8c7e4ab8fa20259a57`) |
| `llama.cpp-fork/patches/graded-moe-g1.patch` | the G1 update on top (26 files, +786 lines) |
| [`llama-fork-vulkan-graded-g1-win-x64.zip`](https://github.com/wuliaott/moe-expert-grading/releases/download/v1.1-graded-moe/llama-fork-vulkan-graded-g1-win-x64.zip) | Windows x64 binaries with G1 + MTP fix (release asset) |
| `llama.cpp-fork/quant_recipe.py` | one-command quantization, dry-run by default |
| `llama.cpp-fork/split_moe_experts.py` | expert tiering + `moe_expert_groups` |
| `llama.cpp-fork/fix_imatrix_names.py` | tier imatrix aliases |
| `llama.cpp-fork/README-TECH.md` | full write-up: pitfalls, storage, reproduction |

MIT, same as upstream.
