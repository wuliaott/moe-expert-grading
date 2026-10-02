#!/usr/bin/env python3
"""
Slice an imatrix per expert tier so llama-quantize accepts a split MoE model.

Why
---
llama-quantize validates an importance-matrix entry against

    imatrix.size() == tensor.ne[0] * tensor.ne[2]        (src/llama-quant.cpp)

and aborts the whole run on a mismatch (token_embd excepted).

For an unsplit expert tensor the imatrix is  ne0 * n_expert,
e.g. down_exps [512, 2048, 256] -> 131072 values.
After split_moe_experts.py each tier tensor is  ne0 * n_tier,
e.g. down_exps_cold [512, 2048, 128] -> 65536 values.
The original entry no longer fits, so the run dies on the first expert tensor.

This script emits one imatrix entry per tier, sliced along the expert axis with
the same ranking the split used, named exactly like the split tensors:

    blk.3.ffn_down_exps_cold.weight.in_sum2
    blk.3.ffn_down_exps_cold.weight.counts

Usage:
  python fix_imatrix_names.py imatrix.gguf tiers_per_layer.json out.gguf
"""
import json
import os
import sys

import numpy as np

try:
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
except ImportError:
    sys.exit("need: pip install gguf numpy")

SUFFIXES = ("hot", "warm", "cold")
NAMES = ("gate", "up", "gate_up", "down")


def log(m):
    print(m, flush=True)


def load_groups(tiers_path, n_expert):
    """{layer: {tier: [expert ids]}} from tiers_per_layer.json."""
    with open(tiers_path, encoding="utf-8") as f:
        data = json.load(f)

    n_hot = int(round(n_expert * float(data.get("hot_frac", 0.20))))
    n_warm = int(round(n_expert * float(data.get("warm_frac", 0.30))))

    out = {}
    for lkey, entry in (data.get("layers") or {}).items():
        order = entry.get("expert_order_by_energy")
        if not order or len(order) != n_expert:
            continue
        out[int(lkey)] = {
            # split_moe_experts.py writes expert slabs with np.nonzero(), i.e.
            # ascending expert id. The imatrix rows must follow that same order,
            # otherwise each expert quantizes with another expert's importance.
            "hot":  sorted(int(x) for x in order[:n_hot]),
            "warm": sorted(int(x) for x in order[n_hot:n_hot + n_warm]),
            "cold": sorted(int(x) for x in order[n_hot + n_warm:]),
        }
    log(f"tiers: {len(out)} layers, hot {n_hot} / warm {n_warm} / "
        f"cold {n_expert - n_hot - n_warm}")
    return out


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    src, tiers_path, dst = sys.argv[1], sys.argv[2], sys.argv[3]

    rdr = GGUFReader(src, "r")

    kv, ktype = {}, {}
    for k in rdr.fields.keys():
        key = str(k)
        if key.startswith("GGUF."):
            continue
        try:
            fld = rdr.get_field(key)
            kv[key] = fld.contents()
            ktype[key] = fld.types[-1] if getattr(fld, "types", None) else None
        except Exception:
            kv[key] = None
            ktype[key] = None

    arch = None
    for k, v in kv.items():
        if k.endswith("general.architecture") and isinstance(v, str):
            arch = v
    log(f"source arch = {arch or '(legacy, none stored)'}")

    n_expert = 0
    for k, v in kv.items():
        if k.endswith("expert_count"):
            n_expert = int(v)

    # fall back to the ranking file: expert_order_by_energy has one id per expert
    if n_expert <= 0:
        with open(tiers_path, encoding="utf-8") as f:
            _d = json.load(f)
        for _e in (_d.get("layers") or {}).values():
            _o = _e.get("expert_order_by_energy")
            if _o:
                n_expert = len(_o)
                break
        log(f"expert_count absent from metadata; using {n_expert} from "
            f"{os.path.basename(tiers_path)}")

    groups = load_groups(tiers_path, n_expert)
    if not groups:
        sys.exit(f"no usable layer rankings in {tiers_path} for "
                 f"n_expert={n_expert}")

    w = GGUFWriter(dst, arch or "unknown")
    for key, arr in kv.items():
        vt = ktype[key]
        if isinstance(arr, (list, tuple)) and arr and isinstance(arr[0], str):
            w.add_array(key, [str(x) for x in arr])
        elif isinstance(arr, bool):
            w.add_key_value(key, arr, GGUFValueType.BOOL)
        elif isinstance(arr, int):
            w.add_key_value(key, arr, vt or GGUFValueType.UINT32)
        elif isinstance(arr, float):
            w.add_key_value(key, arr, GGUFValueType.FLOAT32)
        elif isinstance(arr, str):
            w.add_key_value(key, arr, GGUFValueType.STRING)
        elif isinstance(arr, (list, tuple)):
            w.add_array(key, list(arr))

    by_name = {t.name: t for t in rdr.tensors}
    n_sliced = n_copied = 0
    per_layer = {}

    for t in rdr.tensors:
        name = t.name
        arr = np.asarray(t.data)
        shape = [int(x) for x in t.shape]

        if "_exps.weight.in_sum2" not in name:
            if "_exps.weight.counts" in name:
                continue          # written alongside its in_sum2 below
            w.add_tensor(name, arr, raw_shape=shape)
            n_copied += 1
            continue

        p = name.split(".")
        il = int(p[1])
        stem = p[2]                       # e.g. "ffn_gate_exps"
        proj = stem[len("ffn_"):-len("_exps")] if stem.startswith("ffn_") \
            and stem.endswith("_exps") else ""
        if proj not in NAMES or il not in groups:
            w.add_tensor(name, arr, raw_shape=shape)
            n_copied += 1
            continue

        # llama.cpp validates imatrix.size() == tensor.ne[0] * tensor.ne[2], so
        # the entry is a FLAT run of n_expert * ne0 values with the expert axis
        # outermost (ggml ne0 is the fastest axis). Reshape to [n_expert, ne0].
        ne0 = shape[0]
        if ne0 * n_expert != arr.size:
            ne0 = arr.size // n_expert
        mat = arr.reshape(n_expert, ne0)
        if mat.shape[0] != n_expert:
            log(f"  WARNING {name}: {mat.shape[0]} rows != {n_expert}; "
                "copying unchanged")
            w.add_tensor(name, arr, raw_shape=shape)
            n_copied += 1
            continue

        base = f"blk.{il}.ffn_{proj}_exps"
        per_layer.setdefault(il, {})[proj] = len(groups[il]["cold"])

        for tier in SUFFIXES:
            ids = groups[il][tier]
            sub = mat[ids, :]
            w.add_tensor(f"{base}_{tier}.weight.in_sum2",
                         sub.reshape(-1), raw_shape=[ne0, len(ids)])
            cnt = by_name.get(f"{base}.weight.counts")
            if cnt is not None:
                # counts may arrive as [n_expert] or [1, n_expert]
                c = np.asarray(cnt.data).reshape(-1)
                if c.size != n_expert:
                    log(f"  WARNING counts {base}: {c.size} != {n_expert}")
                    w.add_tensor(f"{base}_{tier}.weight.counts",
                                 c[:len(ids)], raw_shape=[len(ids)])
                else:
                    w.add_tensor(f"{base}_{tier}.weight.counts",
                                 c[ids], raw_shape=[len(ids)])
            n_sliced += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    log(f"wrote {dst}")
    log(f"  {n_sliced} tier entries sliced, {n_copied} copied verbatim")
    for il in sorted(per_layer)[:3]:
        log(f"  layer {il}: {per_layer[il]}")
    log("  the quantize log must NOT print 'imatrix size' or "
        "'did not find weights'")


if __name__ == "__main__":
    main()