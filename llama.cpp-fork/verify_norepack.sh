#!/bin/bash
# verify_norepack.sh
#
# Hypothesis: ggml's runtime weight repack only ships traits for
# Q1_0/Q2_K/Q4_0/Q4_K/Q5_K/Q6_K/IQ4_NL/MXFP4/Q8_0 -- NOT Q3_K -- and its
# shape gate checks only ne[1] % 8, never ne[2] (the expert count).
#
#   Q3_K  -> not repacked -> normal vec_dot/tiled path -> PPL 1.5180 (good)
#   Q4_K  -> repacked     -> repack forward_mul_mat_id -> PPL 3.7313 (broken)
#   upstream Q4_K_S        -> repacked, ne[2]=256 (256%8==0) -> 1.5167 (good)
#   ours v9                -> repacked, ne[2]=51  (51%8==3)  -> 3.7313 (bad)
#
# Rebuild with -DGGML_CPU_REPACK=OFF into a separate build dir and re-run the
# SAME two models. If v9 recovers to ~1.52 while v5b stays ~1.518, the repack
# kernel (or its interaction with our grouped mul_mat_id on non-multiple-of-8
# expert counts) is the culprit.
#
# usage: nohup bash /mnt/workspace/verify_norepack.sh > /root/nr_pipe.log 2>&1 &
# watch: tail -f /root/nr_pipe.log

set -u
set -x

SRC=/root/llama.cpp-master
BLD=/root/build-norepack
CORPUS=/mnt/workspace/ppl_small.txt
TH=20

echo "=== [1/4] cmake configure (GGML_CPU_REPACK=OFF) ==="
cmake -G Ninja -B "$BLD" -S "$SRC" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_VULKAN=OFF -DGGML_CUDA=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DGGML_CPU_REPACK=OFF > /root/nr_cmake.log 2>&1
echo "rc_cmake=$?"

grep -E "GGML_CPU_REPACK" /root/nr_cmake.log || true

echo "=== [2/4] build llama-perplexity (no-repack) ==="
ninja -C "$BLD" llama-perplexity > /root/nr_build.log 2>&1
echo "rc_build=$?"
ls -lh "$BLD"/bin/llama-perplexity
# confirm the macro really is gone
grep -c "GGML_USE_CPU_REPACK" /root/nr_build.log || true

echo "=== [3/4] PPL on graded-v9 (Q4_K hot tier) WITHOUT repack ==="
"$BLD"/bin/llama-perplexity -m /root/graded-v9.gguf -f "$CORPUS" -t "$TH" > /root/nr_v9.log 2>&1
echo "rc_v9=$?"
grep "Final estimate" /root/nr_v9.log

echo "=== [4/4] PPL on graded-v5b (all Q3_K) WITHOUT repack, control ==="
"$BLD"/bin/llama-perplexity -m /root/graded-v5b.gguf -f "$CORPUS" -t "$TH" > /root/nr_v5b.log 2>&1
echo "rc_v5b=$?"
grep "Final estimate" /root/nr_v5b.log

echo ""
echo "=== SUMMARY (repack OFF) ==="
echo -n "v9  (hot Q4_K)     : "; grep -o "PPL = [0-9.]*" /root/nr_v9.log | tail -1
echo -n "v5b (all Q3_K)     : "; grep -o "PPL = [0-9.]*" /root/nr_v5b.log | tail -1
echo ""
echo "=== REFERENCE (repack ON, same models) ==="
echo "v9  = 3.7313   (broken)"
echo "v5b = 1.5180   (good)"
echo "upstream merged Q4_K_S = 1.5167 | Q3_K_S = 1.5404 | FP16 = 1.5053"
echo ""
echo "VERDICT:"
echo "  v9 ~= 1.52 and v5b ~= 1.518  -> repack kernel IS the culprit"
echo "  v9 still >= 3                 -> repack exonerated, look elsewhere"
echo DONE_NOREPACK
