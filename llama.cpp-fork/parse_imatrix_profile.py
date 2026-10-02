#!/usr/bin/env python3
"""
Parse a llama.cpp imatrix .gguf and extract the PER-EXPERT activation profile.

LAYOUT -- verified against a real Qwen3.6-35B-A3B imatrix (not assumed):

    blk.N.ffn_gate_exps.weight.in_sum2   ggml shape [2048, 256]
                                         numpy shape (256, 2048)  <- column-major
    blk.N.ffn_down_exps.weight.in_sum2   ggml shape [ 512, 256]

  ne[0] = 2048 = dense dim (n_embd * 2 for merged gate_up, or n_ff)
  ne[1] =  256 = n_expert            <-- THE EXPERT AXIS IS ne[1]

So t.data in numpy is (n_expert, dense_dim). Sum over axis 1 to get one
activation-energy number per expert. There is NO top-k axis in the tensor;
top-k is applied later at quantize time, which is exactly what we want to
measure: how much energy each expert draws when it IS selected.

.counts is a separate tensor and carries no expert detail for _exps rows, so it
is deliberately ignored.

HISTORY -- an earlier version assumed [n_expert_used, n_expert, n_row] and
reported 2048 "experts" for a 256-expert model, flattening the distribution
(Gini 0.24). The self-test had been generated with the same wrong assumption,
so it could not catch it. Only a real imatrix did.
"""
import json
import math
import os
import sys
from collections import defaultdict
from statistics import NormalDist

import numpy as np
from gguf import GGUFReader

NORMAL = NormalDist()


def gini(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[x >= 0]
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def norm_entropy(x):
    x = np.asarray(x, dtype=np.float64)
    if x.sum() <= 0:
        return float("nan")
    p = x / x.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / math.log2(len(x)))


def lognormal_top_mass(sigma, p):
    """Fraction of total mass held by the top p fraction of a lognormal."""
    u = NORMAL.inv_cdf(1 - p)
    return 0.5 * (1 + math.erf((sigma - u) / math.sqrt(2)))


def main():
    if len(sys.argv) < 2:
        print("usage: parse_imatrix_profile.py <imatrix.gguf> [outdir] "
              "[--hot 0.20] [--warm 0.30] [--n-expert 256]", file=sys.stderr)
        return 2
    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(path))
    HOT = WARM = None
    args_n_expert = None
    for i, a in enumerate(sys.argv):
        if a == "--hot" and i + 1 < len(sys.argv):
            HOT = float(sys.argv[i + 1])
        if a == "--warm" and i + 1 < len(sys.argv):
            WARM = float(sys.argv[i + 1])
        if a == "--n-expert" and i + 1 < len(sys.argv):
            args_n_expert = int(sys.argv[i + 1])
    HOT = 0.20 if HOT is None else HOT
    WARM = 0.30 if WARM is None else WARM
    os.makedirs(outdir, exist_ok=True)

    print(f"reading {path} ...", flush=True)
    rdr = GGUFReader(path, "r")

    # ---- pick the expert axis without assuming an ordering ----------------
    # gate/up carry the embedding dim (2048) and down carries n_ff_exp (512),
    # so the DENSE axis changes between projections while the EXPERT axis is
    # constant. Identify the constant axis instead of trusting position: the
    # gguf writer and the reader disagree about axis order, and an earlier
    # hard-coded ne[1] assumption silently read this model as 2048 experts.
    # ---- identify the expert axis from the file, never from position ----
    # gate/up carry the embedding dim; down carries n_ff_exp (different value).
    # The axis that stays CONSTANT across gate/up/down is the expert axis.
    # Position cannot be trusted: the writer and reader disagree about axis
    # order, and a hard-coded ne[1] read this very model as 2048 experts
    # instead of 256, flattening the entire distribution.
    gu_shapes, dn_shapes = set(), set()
    for t in rdr.tensors:
        if "_exps.weight.in_sum2" not in t.name:
            continue
        ne = [int(x) for x in t.shape]
        if len(ne) != 2:
            continue
        (dn_shapes if "down" in t.name else gu_shapes).add(tuple(ne))

    # gate/up dense dim != down dense dim, so compare ACROSS the two classes:
    # whichever axis has different values in the two classes is the dense axis,
    # and the other one is the expert axis. Comparing only within gate/up (both
    # share one dense value) makes BOTH axes look constant and picks the wrong
    # one -- that is exactly how this model first came out as 2048 experts.
    gu_vals = sorted({s[0] for s in gu_shapes} | {s[1] for s in gu_shapes})
    dn_vals = sorted({s[0] for s in dn_shapes} | {s[1] for s in dn_shapes})
    exp_axis, basis = None, ""
    if gu_shapes and dn_shapes and len(gu_shapes) == len(dn_shapes) == 1:
        g, d = next(iter(gu_shapes)), next(iter(dn_shapes))
        # find the axis where gate/up and down DISAGREE -> that is dense
        dense_axis = next((ax for ax in (0, 1) if g[ax] != d[ax]), None)
        if dense_axis is not None:
            exp_axis = 1 - dense_axis
            basis = (f"gate/up ne={list(g)} vs down ne={list(d)}: ne[{dense_axis}] "
                     f"differs ({g[dense_axis]} vs {d[dense_axis]}) = dense axis, "
                     f"so ne[{exp_axis}]={g[exp_axis]} = expert axis")
    if exp_axis is None and args_n_expert is not None:
        exp_axis = next((ax for ax in (0, 1)
                         if all(args_n_expert == s[ax] for s in gu_shapes)), None)
        basis = f"from --n-expert {args_n_expert} (cross-projection check failed)"
    if exp_axis is None:
        exp_axis, basis = 1, "FALLBACK ne[1] -- VERIFY n_expert!"

    n_expert = next((s[exp_axis] for s in sorted(gu_shapes)), None)
    if n_expert is None:
        print("ERROR: could not determine n_expert from tensor shapes.", file=sys.stderr)
        return 1

    print(f"expert axis = ne[{exp_axis}]  -> n_expert = {n_expert}")
    print(f"  basis        : {basis}")
    print(f"  gate/up shape: {sorted(gu_shapes)}")
    print(f"  down    shape: {sorted(dn_shapes)}")
    if args_n_expert and args_n_expert != n_expert:
        print(f"  WARNING: --n-expert {args_n_expert} != file {n_expert}; trusting file.")

    per_layer = defaultdict(lambda: defaultdict(float))
    parsed = 0
    skipped = []

    for t in rdr.tensors:
        name = t.name
        if "_exps.weight.in_sum2" not in name:
            continue
        if "down" in name:
            continue  # gate/up are the cleaner router proxies

        ne = [int(x) for x in t.shape]
        if len(ne) != 2:
            skipped.append((name, ne))
            continue
        n_exp = ne[exp_axis]
        dense = ne[1 - exp_axis]
        arr = np.asarray(t.data, dtype=np.float64).reshape(n_exp, dense)

        if n_exp != n_expert:
            skipped.append((name, ne))
            continue

        layer = -1
        if "blk." in name:
            try:
                layer = int(name.split("blk.")[1].split(".")[0])
            except (IndexError, ValueError):
                layer = -1

        for e, v in enumerate(arr.sum(axis=1)):
            per_layer[layer][e] += float(v)
        parsed += 1

    print(f"per-expert tensors parsed : {parsed}")
    if skipped:
        print(f"skipped (unexpected shape): {skipped[:4]}")
    if not per_layer:
        print("ERROR: no _exps.weight.in_sum2 tensors found.", file=sys.stderr)
        return 1

    layers = sorted(per_layer)
    print(f"layers covered            : {len(layers)}  -> {layers[0]}..{layers[-1]}")
    print(f"experts per layer         : {n_expert}")

    # ---------- global ----------
    glob = np.zeros(n_expert)
    for L in layers:
        for e, v in per_layer[L].items():
            glob[e] += v
    if glob.sum() <= 0:
        print("ERROR: zero total activation energy.", file=sys.stderr)
        return 1
    glob /= glob.sum()

    order = np.argsort(-glob)
    cum = np.cumsum(glob[order])

    print("\n" + "=" * 70)
    print("MEASURED global activation curve (all layers pooled)")
    print("=" * 70)
    print(f"{'top X%':>8} {'experts':>8} {'cum. energy':>12}")
    curve = {}
    for pct in (1, 5, 10, 20, 25, 30, 40, 50, 60, 75, 90, 100):
        k = max(1, int(round(n_expert * pct / 100)))
        curve[pct] = float(cum[k - 1])
        print(f"{pct:>7}% {k:>8} {cum[k-1]*100:>11.2f}%")

    g = gini(glob)
    print(f"\nGini              : {g:.4f}")
    print(f"normalized entropy: {norm_entropy(glob):.4f}   (1.0 = uniform)")
    print(f"experts >0.01% mass: {(glob > 0.0001).sum()} / {n_expert}")
    print(f"experts  0 mass    : {(glob == 0).sum()} / {n_expert}")
    print(f"hottest expert     : {glob.max()*100:.3f}% of all energy "
          f"(uniform would be {100/n_expert:.3f}%)")

    # ---------- hypotheses ----------
    if not math.isnan(g):
        sigma = NORMAL.inv_cdf((1 + g) / 2) * math.sqrt(2)
        print("\n" + "=" * 70)
        print("vs the two hypotheses in EXPERT_ACTIVATION_PROFILE.md")
        print("=" * 70)
        print(f"measured Gini {g:.4f} -> lognormal sigma = {sigma:.4f}")
        print(f"{'top X%':>8} {'measured':>10} {'B lognormal':>13} {'A powerlaw':>12}")
        A = {20: .800, 30: .854, 50: .918}
        for pct in (10, 20, 30, 50, 75):
            ma = A.get(pct)
            print(f"{pct:>7}% {curve[pct]*100:>9.2f}% "
                  f"{lognormal_top_mass(sigma, pct/100)*100:>12.2f}%"
                  + (f" {ma*100:>11.2f}%" if ma else f" {'-':>12}"))

    # ---------- per-layer ----------
    k_h = int(round(n_expert * HOT))
    k_w = int(round(n_expert * (HOT + WARM)))
    tiers = {}
    gs = []
    print("\n" + "=" * 70)
    print(f"per-layer tiers (hot {HOT:.0%} = {k_h} | warm {WARM:.0%} = {k_w-k_h} | "
          f"cold = {n_expert-k_w})")
    print("=" * 70)
    print(f"{'layer':>6} {'Gini':>7} {'nrmH':>6} {'hot%':>8} {'warm%':>8} "
          f"{'cold%':>8} {'zero':>6}")

    for L in layers:
        v = np.zeros(n_expert)
        for e, x in per_layer[L].items():
            v[e] = x
        if v.sum() <= 0:
            continue
        v /= v.sum()
        c = np.cumsum(np.sort(v)[::-1])
        hot_m, warm_m = float(c[k_h - 1]), float(c[k_w - 1])
        gv = gini(v)
        gs.append(gv)
        print(f"{L:>6} {gv:>7.4f} {norm_entropy(v):>6.3f} "
              f"{hot_m*100:>7.2f}% {(warm_m-hot_m)*100:>7.2f}% "
              f"{(1-warm_m)*100:>7.2f}% {(v==0).sum():>6}")
        tiers[L] = {"hot": k_h, "warm": k_w - k_h, "cold": n_expert - k_w,
                    "hot_mass": hot_m, "warm_mass": warm_m, "gini": gv,
                    "zero_experts": int((v == 0).sum()),
                    "expert_order_by_energy": [int(e) for e in np.argsort(-v)]}

    if gs:
        print(f"\nper-layer Gini: min {min(gs):.4f}  max {max(gs):.4f}  "
              f"spread {max(gs)-min(gs):.4f}  mean {sum(gs)/len(gs):.4f}")
        if max(gs) - min(gs) > 0.05:
            print("-> spread > 0.05: layers differ materially. Rank experts")
            print("   PER LAYER; do not reuse one global expert list.")

    # ---------- outputs ----------
    with open(os.path.join(outdir, "curve_global.json"), "w") as f:
        json.dump({"n_expert": int(n_expert), "n_layers": len(layers),
                   "gini": g, "norm_entropy": norm_entropy(glob),
                   "curve_pct_to_cum_mass": curve,
                   "expert_order_by_energy": [int(e) for e in order],
                   "expert_energy": glob.tolist()}, f, indent=2)
    with open(os.path.join(outdir, "tiers_per_layer.json"), "w") as f:
        json.dump({"hot_frac": HOT, "warm_frac": WARM,
                   "n_expert": int(n_expert), "layers": tiers}, f, indent=2)

    print(f"\nwrote {outdir}/curve_global.json")
    print(f"wrote {outdir}/tiers_per_layer.json")

    # ---------- verdict ----------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    h20 = curve[20]
    print(f"top 20% of experts carry {h20*100:.2f}% of activation energy")
    if h20 >= 0.90:
        print("-> STEEP. Cold half is near-dead weight; IQ3_XXS is safe and")
        print("   IQ2_XXS becomes worth a test.")
    elif h20 >= 0.75:
        print("-> MODERATE. IQ3_XXS is the right cold tier (matches APEX Compact).")
    elif h20 >= 0.55:
        print("-> MODERATELY FLAT. Cold tier at IQ3_M; do not drop to 2-bit.")
        print("   The hot-tier premium buys less here -- reconsider whether a")
        print("   3-tier expert split earns its engineering cost.")
    else:
        print("-> FLAT. A 3-tier expert split is unlikely to pay off. Verify first:")
        print("   rerun with more chunks, and compare against a general-corpus")
        print("   imatrix to see how much of this flatness is the agentic corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())