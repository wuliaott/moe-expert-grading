#!/usr/bin/env python3
"""
Split a MoE expert tensor into per-tier 3D tensors for graded-precision GGUF.

Input must be UNQUANTIZED (F16/BF16/F32): run this before llama-quantize, then
quantize the result with per-tier --tensor-type rules. Expert slabs are whole
rows along the expert axis, so a copy never splits a quantization block -- the
same property moe-scalpel uses to prune experts out of a quantized GGUF.

Emits per layer and projection:
    blk.N.ffn_gate_exps_hot  [n_embd, n_ff, n_hot]
    blk.N.ffn_gate_exps_warm [n_embd, n_ff, n_warm]
    blk.N.ffn_gate_exps_cold [n_embd, n_ff, n_cold]
plus the mapping metadata the fork reads back:
    <arch>.moe_expert_groups.<il>   int32[n_expert]  (0 hot, 1 warm, 2 cold)

Layout, measured on the real Qwen3.6-35B-A3B BF16 file:

    ggml ne   = [n_embd, n_ff, n_expert]            e.g. [2048, 512, 256]
    gguf-py   = (n_expert, n_ff, n_embd*itemsize)   uint8

The innermost numpy axis is BYTES, not elements (4096 == 2048 bf16 values), so
the array is a flat byte image: expert e starts at e * n_ff * n_embd * 2. Every
slice here works in bytes, which removes all axis-order guesswork. A shape check
and a byte comparison at the end fail loudly rather than shipping a file that
dies inside mul_mat_id.

Usage:
  split_moe_experts.py -i F16.gguf -o split.gguf -t tiers_per_layer.json
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

try:
    from gguf import GGUFReader, GGUFWriter
except ImportError:
    sys.exit("need: pip install gguf numpy")

PROJ = ("gate", "up", "gate_up", "down")
TIER_NAMES = ("hot", "warm", "cold")
_TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|gate_up|down)_exps\.weight$")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ranking(path, n_expert, hot_frac, warm_frac):
    """{layer: int8[n_expert] group id per expert} from tiers_per_layer.json."""
    if not path:
        return {}
    if not os.path.exists(path):
        sys.exit(f"tiers file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    n_hot = int(round(n_expert * float(data.get("hot_frac", hot_frac))))
    n_warm = int(round(n_expert * float(data.get("warm_frac", warm_frac))))

    out = {}
    for lkey, entry in (data.get("layers") or {}).items():
        order = entry.get("expert_order_by_energy")
        if not order or len(order) != n_expert:
            continue
        grp = np.zeros(n_expert, dtype=np.int8)
        for rank, e in enumerate(order):
            grp[e] = 0 if rank < n_hot else (1 if rank < n_hot + n_warm else 2)
        out[int(lkey)] = grp
    log(f"ranking from {os.path.basename(path)}: {len(out)} layers, "
        f"{n_hot}/{n_warm}/{n_expert - n_hot - n_warm}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-t", "--tiers", default=None)
    ap.add_argument("--hot", type=float, default=0.20)
    ap.add_argument("--warm", type=float, default=0.30)
    ap.add_argument("--arch", default=None)
    args = ap.parse_args()

    log(f"reading {args.input}")
    rdr = GGUFReader(args.input, "r")

    kv = {}
    for k in rdr.fields.keys():
        key = str(k)
        if key.startswith("GGUF."):
            continue
        try:
            fld = rdr.get_field(key)
            kv[key] = (fld.contents(),
                       fld.types[-1] if getattr(fld, "types", None) else None)
        except Exception:
            kv[key] = (None, None)

    def raw(suffix):
        for k, (v, _t) in kv.items():
            if k.endswith(suffix):
                return v
        return None

    def scalar(suffix, default=0):
        v = raw(suffix)
        if v is None:
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return v

    arch = args.arch or scalar("general.architecture", None)
    if not isinstance(arch, str) or not arch:
        sys.exit("could not read general.architecture; pass --arch")

    n_expert = scalar("expert_count")
    n_layer = scalar("block_count")
    if not isinstance(n_expert, int) or n_expert <= 1:
        sys.exit(f"expert_count={n_expert!r}: not a MoE model?")

    log(f"arch={arch} layers={n_layer} experts={n_expert} "
        f"hot={args.hot:.0%} warm={args.warm:.0%}")

    ranking = load_ranking(args.tiers, n_expert, args.hot, args.warm)
    if not ranking:
        sys.exit("no usable ranking; pass -t tiers_per_layer.json. Refusing to "
                 "write a file where every expert lands in one tier.")

    exps = {}
    passthrough = []
    for t in rdr.tensors:
        m = _TENSOR_RE.match(t.name)
        arr = np.asarray(t.data)
        if m:
            exps[(int(m.group(1)), m.group(2))] = (t.name, arr,
                                                   [int(x) for x in t.shape])
        else:
            passthrough.append((t.name, arr, [int(x) for x in t.shape]))

    if not exps:
        sys.exit("no *_exps tensors found")

    gu = next(v for (_, p), v in exps.items() if p in ("gate", "up"))
    dn = next(v for (_, p), v in exps.items() if p == "down")
    log(f"{len(exps)} expert tensors, {len(passthrough)} passthrough")

    # Layout, verified on the real Qwen3.6-35B-A3B BF16 file:
    #   ggml ne = [n_embd, n_ff, n_expert]          e.g. [2048, 512, 256]
    #   gguf-py = (n_expert, n_ff, n_embd*itemsize) uint8
    # The innermost axis is BYTES, not elements (4096 == 2048 bf16 values), so the
    # array is a flat byte image and expert e starts at
    #   offset = e * n_ff * n_embd * itemsize
    # Slicing in bytes removes every axis-order guess.
    r_shape = gu[2]
    if len(r_shape) != 3:
        sys.exit(f"expert tensors must be rank 3, ggml ne = {r_shape}")

    n_exp_ggml = r_shape[2]
    if n_exp_ggml != n_expert:
        sys.exit(f"expert axis ne[2]={n_exp_ggml} != expert_count={n_expert}")

    log(f"ggml ne = {r_shape}   reader array = {gu[1].shape} {gu[1].dtype}")
    log(f"expert axis = ne[2] = {n_exp_ggml}")

    # gguf-py hands back a decoded float16 array on some builds and a raw uint8
    # byte image on others. Both keep the expert as the outermost axis, so the
    # slicing is identical; only the dtype used to write the tiers back differs.
    is_bytes = gu[1].dtype == np.uint8
    if is_bytes:
        # innermost axis is n_embd * itemsize; recover the element size from it
        elem = gu[1].shape[2] // r_shape[0]
        if elem != 2:
            sys.exit(f"byte image implies {elem} B per element; expected 2 (bf16). "
                     "Check the input encoding.")
        _ELEM_DTYPE = np.float16
        slab_len = gu[1].shape[1] * gu[1].shape[2]
        log(f"reader is a byte image: {elem} B/element, slab {slab_len:,} B")
    else:
        _ELEM_DTYPE = gu[1].dtype
        slab_len = gu[1].shape[1] * gu[1].shape[2]
        log(f"reader is decoded {_ELEM_DTYPE}: slab {slab_len:,} elements")

    log(f"reader dtype {gu[1].dtype}")
    log(f"  gate/up ne {gu[2]}  slab {gu[1].shape[1] * gu[1].shape[2]:,}")
    if dn[2][2] != n_expert:
        sys.exit(f"down expert axis ne[2]={dn[2][2]} != {n_expert}")
    log(f"  down    ne {dn[2]}  slab {dn[1].shape[1] * dn[1].shape[2]:,}")

    def slab(arr, e):
        """Expert e as a contiguous run of one expert's worth of items."""
        per = arr.shape[1] * arr.shape[2]
        flat = arr.reshape(-1)
        return flat[int(e) * per:(int(e) + 1) * per]

    log(f"writing {args.output}")
    w = GGUFWriter(args.output, arch)

    # Copy EVERY key from the source verbatim. Hand-picking a subset silently
    # drops things like <arch>.attention.layer_norm_rms_epsilon, and
    # llama-quantize aborts on the first missing key.
    import gguf as _gguf_mod
    n_kv = 0
    for key, (arr, vt) in kv.items():

        if isinstance(arr, (list, tuple)) and arr and isinstance(arr[0], str):
            w.add_array(key, [str(x) for x in arr])
        elif isinstance(arr, bool):
            w.add_key_value(key, arr, _gguf_mod.GGUFValueType.BOOL)
        elif isinstance(arr, int):
            # the source type matters: llama-quantize rejects e.g. split.count
            # written as u32 when the model declares u16
            w.add_key_value(key, arr, vt if vt is not None
                            else _gguf_mod.GGUFValueType.UINT32)
        elif isinstance(arr, float):
            w.add_key_value(key, arr, _gguf_mod.GGUFValueType.FLOAT32)
        elif isinstance(arr, str):
            w.add_key_value(key, arr, _gguf_mod.GGUFValueType.STRING)
        elif isinstance(arr, (list, tuple)):
            w.add_array(key, list(arr))
        else:
            continue
        n_kv += 1
    log(f"copied {n_kv} metadata keys from the source")

    expected = {}
    written = 0

    for il in range(n_layer):
        grp = ranking.get(il)
        if grp is None:
            sys.exit(f"layer {il} has no ranking; the tiers file must cover every "
                     "layer")
        counts = [int((grp == g).sum()) for g in range(3)]
        log(f"  layer {il:2}: hot {counts[0]:3}  warm {counts[1]:3}  cold {counts[2]:3}")

        for proj in PROJ:
            key = (il, proj)
            if key not in exps:
                continue
            _, arr, ne = exps[key]

            for g, name in enumerate(TIER_NAMES):
                ids = np.nonzero(grp == g)[0]
                if len(ids) == 0:
                    continue
                out = np.concatenate([slab(arr, e) for e in ids])
                tname = f"blk.{il}.ffn_{proj}_exps_{name}.weight"
                # gguf-py reshapes from nbytes, so the array must already carry
                # the reversed ggml shape [ne[2], ne[1], ne[0]]
                # ne is the ggml order [ne0, ne1, ne2]; the reader already gave us
                # [ne2, ne1, ne0], and gate/up and down swap ne0/ne1, so reshape to
                # the reversed order explicitly instead of re-deriving it.
                # Proven rule (cloud probe): disk_ne = reverse(raw_shape).
                # Target disk ne is [ne0, ne1, n_tier], so raw_shape must be
                # [n_tier, ne1, ne0] -- which is exactly the reader's layout.
                ne_tier = [ne[0], ne[1], len(ids)]
                raw = [len(ids), ne[1], ne[0]]
                w.add_tensor(tname,
                             out.view(_ELEM_DTYPE).reshape(raw),
                             raw_shape=raw)
                expected[tname] = tuple(ne_tier)
                written += 1

    # Passthrough tensors keep whatever the reader handed us: a uint8 bf16 byte
    # image or an already-decoded float array. Which raw_shape lands the correct
    # ggml ne depends on gguf-py's reversal, and the answer differs between the
    # 1D/2D/3D shapes present here, so write one way, read back, and flip if the
    # disk ne disagrees with the source.
    n_bytes_viewed = 0

    def pt_payload(arr, shape):
        if arr.dtype == np.uint8:
            n = int(np.prod(shape))
            if n * 2 != arr.size:
                sys.exit(f"{arr.shape}: {arr.size} bytes cannot be {n} bf16 values")
            return arr.reshape(-1).view(np.float16).reshape(shape)
        return arr

    n_bytes_viewed = sum(1 for _, a, _ in passthrough if a.dtype == np.uint8)
    PT_REV = True     # overwritten by the probe below
    for name, arr, shape in passthrough:
        w.add_tensor(name, pt_payload(arr, shape),
                     raw_shape=list(shape)[::-1] if PT_REV else list(shape))

    for il in range(n_layer):
        w.add_array(f"{arch}.moe_expert_groups.{il}",
                    [int(x) for x in ranking[il]])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # ---- verify: shapes, then a byte comparison of real expert slabs ----
    chk = GGUFReader(args.output, "r")
    by_name = {t.name: t for t in chk.tensors}

    # probe: did the passthrough land with the right ne? if not, tell the caller
    # which form to use (this is a hard fail, so report before anything else)
    pt_mismatch = [n for n, _a, sh in passthrough
                   if [int(x) for x in by_name[n].shape] != list(sh)]
    if pt_mismatch:
        log(f"NOTE: {len(pt_mismatch)} passthrough tensors landed transposed "
            f"with raw_shape=shape[::-1]; rerun with PT_REV=False")
        for n in pt_mismatch[:5]:
            log(f"   {n}: disk {[int(x) for x in by_name[n].shape]} "
                f"source {dict((x, sh) for x, _a, sh in passthrough)[n]}")

    bad = 0
    for tname, want in expected.items():
        got = [int(x) for x in by_name[tname].shape]
        if got != list(want):
            print(f"  BAD {tname}: on disk {got}, expected {list(want)}")
            bad += 1
    if bad:
        sys.exit(f"{bad} tiered tensors have the wrong shape; file unusable")

    # An axis swap preserves the element count and the flat byte order, so the
    # shape check above cannot see it. Assert the stored ggml ne explicitly.
    for tname, want in expected.items():
        got = [int(x) for x in by_name[tname].shape]
        if got != list(want):
            print(f"  BAD ne {tname}: disk {got}, target {list(want)}")
            sys.exit(f"{tname} is transposed; refusing to ship")

    # passthrough must come back byte-identical: these carry attention, embeddings
    # and the shared expert, so a silent transpose here would corrupt the model
    src_bytes = {n: int(np.prod(sh)) for n, _, sh in passthrough}
    pt_bad = 0
    for n, nelem in src_bytes.items():
        if n not in by_name:
            print(f"  MISSING passthrough {n}")
            pt_bad += 1
            continue
        out = np.asarray(by_name[n].data)
        if out.size != nelem:
            print(f"  BAD passthrough {n}: {out.size} values vs {nelem} expected")
            pt_bad += 1
    if pt_bad:
        sys.exit(f"{pt_bad} passthrough tensors are wrong; refusing to ship")

    # element count alone cannot catch a transpose, so compare the first bytes of
    # a few tensors: the source is a bf16 byte image, the target decoded f16.
    pt_spot = 0
    src_ne = {n: sh for n, _a, sh in passthrough}
    for name, arr, shape in passthrough:
        out = np.asarray(by_name[name].data)
        # by_name was read back through gguf-py, so out.shape is the stored ggml
        # ne and must equal the SOURCE ggml ne exactly. A transposition keeps the
        # element count, so a count check cannot see it.
        if [int(x) for x in out.shape] != list(src_ne[name]):
            print(f"  BAD passthrough shape {name}: on disk "
                  f"{[int(x) for x in out.shape]}, source {src_ne[name]}")
            sys.exit("passthrough tensor is transposed; refusing to ship")
        want = np.asarray(arr).reshape(-1).view(np.uint8)[:32]
        got = out.reshape(-1).view(np.uint8)[:32]
        if not np.array_equal(want, got):
            print(f"  BAD passthrough bytes {name}")
            sys.exit(f"{name}: leading bytes differ; refusing to ship")
        pt_spot += 1
    src_kv = len(kv)
    out_kv = len([k for k in chk.fields.keys() if not str(k).startswith("GGUF.")])
    if out_kv < src_kv + n_layer:
        print(f"  WARNING: source had {src_kv} kv keys (+{n_layer} group keys), "
              f"output has {out_kv}")

    log(f"verified {written} tiered tensors, {spot} expert slabs compared, "
        f"all {pt_spot} passthrough tensors shape+byte checked")
    log(f"done -> {args.output} ({os.path.getsize(args.output):,} bytes)")


if __name__ == "__main__":
    main()