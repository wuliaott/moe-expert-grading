#!/bin/bash
# upstream_baseline.sh
#
# Decide whether "q4_K/q5_K worse than q3_K" comes from upstream/environment
# or from our graded-MoE fork.
#
# It quantizes the MERGED (non-split) F16 model with the official b9561
# binaries and two plain ftype presets, then measures perplexity on the
# same corpus, threads and machine as every graded run.
# No fork code, no tier split, no custom ttypes file is involved.
#
# Judgement:
#   Q4_K_S worse than Q3_K_S  -> upstream/environment decode problem
#   Q4_K_S better than Q3_K_S -> upstream is fine, the fault is in our
#                                grouped mul_mat_id path for high-precision types
#
# usage: nohup bash /mnt/workspace/upstream_baseline.sh > /root/mq_pipe.log 2>&1 &
# watch: tail -f /root/mq_pipe.log

set -u
set -x          # echo every command into the pipe log

B=/mnt/workspace/moe/llama-b9561
IM=/mnt/workspace/moe/agent.imatrix.gguf
SRC=/root/qwen35moe_f16.gguf
CORPUS=/mnt/workspace/ppl_small.txt
TH=20

# b9561 rejects blk.40 on the pure-ftype path:
#   "Bad layer 40 for tensor blk.40.ffn_down_exps.weight. Must be in [0, 40)"
# The per-tensor rule path does NOT run that check (all our graded runs use it).
# blk.40 is the MTP block: llama.cpp logs it as "unused tensor", it never takes
# part in inference, so pin it with one rule and let the ftype preset drive
# every other tensor -> a clean official-Q4_K_S vs official-Q3_K_S baseline.
echo 'blk\.40\..*=q8_0' > /root/ttypes_upstream.txt
cat /root/ttypes_upstream.txt

echo "=== [1/4] quantize Q4_K_S (merged, official b9561) ==="
"$B"/llama-quantize --imatrix "$IM" --tensor-type-file /root/ttypes_upstream.txt "$SRC" /root/m-q4ks.gguf Q4_K_S > /root/q_mq4.log 2>&1
echo "rc_q4_quant=$?  size=$(ls -lh /root/m-q4ks.gguf 2>/dev/null | awk '{print $5}')"

echo "=== [2/4] PPL on Q4_K_S ==="
"$B"/llama-perplexity -m /root/m-q4ks.gguf -f "$CORPUS" -t "$TH" > /root/p_mq4.log 2>&1
echo "rc_q4_ppl=$?"
grep "Final estimate" /root/p_mq4.log

echo "=== [3/4] quantize Q3_K_S (merged, official b9561) ==="
"$B"/llama-quantize --imatrix "$IM" --tensor-type-file /root/ttypes_upstream.txt "$SRC" /root/m-q3ks.gguf Q3_K_S > /root/q_mq3.log 2>&1
echo "rc_q3_quant=$?  size=$(ls -lh /root/m-q3ks.gguf 2>/dev/null | awk '{print $5}')"

echo "=== [4/4] PPL on Q3_K_S ==="
"$B"/llama-perplexity -m /root/m-q3ks.gguf -f "$CORPUS" -t "$TH" > /root/p_mq3.log 2>&1
echo "rc_q3_ppl=$?"
grep "Final estimate" /root/p_mq3.log

echo ""
echo "=== SUMMARY ==="
echo -n "Q4_K_S  PPL: "; grep -o "PPL = [0-9.]*" /root/p_mq4.log | tail -1
echo -n "Q3_K_S  PPL: "; grep -o "PPL = [0-9.]*" /root/p_mq3.log | tail -1
echo -n "FP16 ref   : 1.5053 | graded v5b (q3_K): 1.5180 | graded v8 (q4/q5): 1.6589"
echo ""
echo ALL_DONE_MQ
