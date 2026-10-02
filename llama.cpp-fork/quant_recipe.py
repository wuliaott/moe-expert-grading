#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quant_recipe.py — MoE 分档量化流水线
配方 -> 生成 ttypes 规则 -> 体积预估 -> 量化 -> 体积核对 -> PPL

默认 dry-run：只打印，不写任何文件、不执行量化。
    python3 quant_recipe.py --recipe v8 --target-mib 17673
确认打印结果无误后：
    python3 quant_recipe.py --recipe v8 --target-mib 17673 --run --ppl

体积标定（v4 / v5 两个实测点解出，误差 <= 2 MiB）：
    总 MiB = NON_EXPERT_MIB + N_LAYERS * S / 8
    S = sum_tier (该档专家数 * (bpw_gate + bpw_up + bpw_down))
"""
import argparse
import os
import re
import subprocess
import sys
import time

# ---- 标定常数 ----
NON_EXPERT_MIB = 2996.0          # 非专家张量（attention/norm/router/shexp/embd/output/blk.40）
EXP_PARAMS = 3 * 2048 * 512      # 每专家 3 个投影的参数量（gate/up/down 各 2048x512）

# ---- ggml 类型 bpw ----
BPW = {
    'q2_K': 2.625, 'q3_K': 3.4375, 'q4_K': 4.5, 'q5_K': 5.5,
    'q6_K': 6.5625, 'q8_0': 8.5,
    'iq3_xxs': 3.0625, 'iq4_xs': 4.25, 'iq4_nl': 4.5,
    'f16': 16.0, 'f32': 4.0,
}

PROJ = ('gate', 'up', 'down')
TIER = ('hot', 'warm', 'cold')

# ---- 内置配方 ----
RECIPES = {
    # 当前目标：17.26 GiB，全 K 系列，档内 down 升一档
    'v8': {
        'hot':  {'gate': 'q4_K', 'up': 'q4_K', 'down': 'q5_K'},
        'warm': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q4_K'},
        'cold': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
    },
    # 已实测：15.82 GiB, PPL 1.46（分档结构保留，类型无梯度）
    'v5_uniform': {
        'hot':  {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
        'warm': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
        'cold': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
    },
    # 保守梯度：hot 抬到 q5_K，warm/cold 守 q3_K
    'hot_q5_rest_q3': {
        'hot':  {'gate': 'q5_K', 'up': 'q5_K', 'down': 'q5_K'},
        'warm': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
        'cold': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
    },
    # 压冷：cold gate/up 探 q2_K，down 守 q3_K
    'cold_gu_q2': {
        'hot':  {'gate': 'q5_K', 'up': 'q5_K', 'down': 'q6_K'},
        'warm': {'gate': 'q4_K', 'up': 'q4_K', 'down': 'q4_K'},
        'cold': {'gate': 'q2_K', 'up': 'q2_K', 'down': 'q3_K'},
    },
    # bisect: lift ONLY the hot tier to q4_K, warm/cold stay at the known-good
    # q3_K baseline (v5_uniform, PPL 1.5180). One variable at a time.
    'hot_q4_only': {
        'hot':  {'gate': 'q4_K', 'up': 'q4_K', 'down': 'q4_K'},
        'warm': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
        'cold': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
    },
    # push under APEX's 16.1 GiB: keep hot/warm from hot_q4_only (1.5085 w/o
    # repack) and shave the two least sensitive cold projections to q2_K while
    # cold down holds q3_K -> ~15.59 GiB.
    'v10_cold_q2': {
        'hot':  {'gate': 'q4_K', 'up': 'q4_K', 'down': 'q4_K'},
        'warm': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
        'cold': {'gate': 'q2_K', 'up': 'q2_K', 'down': 'q3_K'},
    },
    # bisect partner: only the hot down projection at q5_K (the 40 q5_K tensors in v8)
    'hot_down_q5_only': {
        'hot':  {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q5_K'},
        'warm': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
        'cold': {'gate': 'q3_K', 'up': 'q3_K', 'down': 'q3_K'},
    },
}


def parse_args():
    ap = argparse.ArgumentParser(
        description='MoE 分档量化：配方 -> 规则 -> 体积预估 -> 量化 -> 核对 -> PPL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--recipe', default='v8', help='内置配方名（--list 查看）')
    ap.add_argument('--list', action='store_true', help='列出内置配方后退出')
    ap.add_argument('--set', action='append', default=[], metavar='TIER.PROJ=TYPE',
                    help='覆盖单档单投影，可重复，如 --set cold.gate=q4_K')
    ap.add_argument('--split', default='/root/split_f16.gguf', help='分档 F16 模型（读层数/档人数）')
    ap.add_argument('--assume', default=None, metavar='hot=51,warm=77,cold=128',
                    help='跳过 split 扫描，直接指定各档专家数')
    ap.add_argument('--assume-layers', type=int, default=40, help='配合 --assume 的层数')
    ap.add_argument('--template', default='/mnt/workspace/moe/ttypes.txt',
                    help='原规则文件，从中继承非分档的 7 条 keep 规则')
    ap.add_argument('--imatrix', default='/mnt/workspace/moe/agent_split.imatrix.gguf')
    ap.add_argument('--quant-bin', default='/mnt/workspace/moe/llama-b9561/llama-quantize')
    ap.add_argument('--rules', default=None, help='规则文件（默认 /mnt/workspace/moe/ttypes_<recipe>.txt）')
    ap.add_argument('--out', default=None, help='输出模型（默认 /root/graded-<recipe>.gguf）')
    ap.add_argument('--log', default=None, help='量化日志（默认 /root/quant_<recipe>.log）')
    ap.add_argument('--target-mib', type=float, default=None, help='目标体积，超差直接失败')
    ap.add_argument('--tol', type=float, default=150.0, help='体积容差 MiB')
    ap.add_argument('--run', action='store_true', help='真正执行量化（缺省只 dry-run）')
    ap.add_argument('--ppl', action='store_true', help='量化后接着跑 PPL')
    # repack bug (see verify_norepack.sh): ggml's runtime weight repack has no
    # Q3_K trait, so Q3_K stays on the normal path (clean) while Q4_K/Q5_K get
    # repacked and computed wrong on our grouped 3D tensors (ne[2] = 51/77).
    # Use the no-repack build until get_optimal_repack_type rejects unsafe shapes.
    ap.add_argument('--ppl-bin', default='/root/build-norepack/bin/llama-perplexity')
    ap.add_argument('--corpus', default='/mnt/workspace/ppl_small.txt')
    ap.add_argument('--threads', type=int, default=20)
    return ap.parse_args()


def die(msg):
    print('ERROR: %s' % msg, file=sys.stderr)
    sys.exit(1)


def scan_split(path):
    """从分档模型读真实层数与各档专家数（ground truth）。"""
    from gguf import GGUFReader
    if not os.path.exists(path):
        die('split 模型不存在: %s' % path)
    r = GGUFReader(path)
    pat = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps_(hot|warm|cold)\.weight$')
    layers, counts, seen = set(), {}, {}
    for t in r.tensors:
        m = pat.match(t.name)
        if not m:
            continue
        il, proj, tier = int(m.group(1)), m.group(2), m.group(3)
        layers.add(il)
        ne2 = int(t.shape[2])          # ggml ne[2] = 该档专家数
        if tier in counts and counts[tier] != ne2:
            die('档 %s 专家数不一致: %d vs %d' % (tier, counts[tier], ne2))
        counts[tier] = ne2
        seen.setdefault(tier, set()).add(proj)
    if not counts:
        die('split 模型里一个 tier 张量都没有，检查路径')
    for tier in TIER:
        if tier not in counts:
            die('split 模型缺少 %s 档' % tier)
        missing = set(PROJ) - seen.get(tier, set())
        if missing:
            die('档 %s 缺少投影: %s' % (tier, sorted(missing)))
    return sorted(layers), counts


def parse_assume(spec, n_layers):
    counts = {}
    for part in spec.split(','):
        k, v = part.split('=')
        counts[k.strip()] = int(v)
    for tier in TIER:
        if tier not in counts:
            die('--assume 缺少 %s' % tier)
    return list(range(n_layers)), counts


def read_keep(template):
    if not os.path.exists(template):
        die('模板规则不存在: %s' % template)
    keep = []
    for line in open(template):
        s = line.strip()
        if not s:
            continue
        if '_exps_hot=' in s or '_exps_warm=' in s or '_exps_cold=' in s:
            continue          # 分档规则，按配方重新生成
        keep.append(s)
    if not keep:
        die('模板里没有非分档规则？检查 --template')
    return keep


def apply_overrides(recipe, overrides):
    for ov in overrides:
        if '=' not in ov:
            die('--set 格式应为 tier.proj=type，收到: %s' % ov)
        key, val = ov.split('=', 1)
        if '.' not in key:
            die('--set 格式应为 tier.proj=type，收到: %s' % ov)
        tier, proj = key.split('.', 1)
        if tier not in recipe:
            die('--set 档名错误: %s（可用 %s）' % (tier, list(recipe)))
        if proj not in recipe[tier]:
            die('--set 投影名错误: %s（可用 %s）' % (proj, list(recipe[tier])))
        if val not in BPW:
            die('--set 类型未知: %s（可用 %s）' % (val, sorted(BPW)))
        recipe[tier][proj] = val
    return recipe


def build_rules(recipe, layers, keep):
    rules = list(keep)
    for il in layers:
        for proj in PROJ:
            for tier in TIER:
                rules.append(r'blk\.%d\.ffn_%s_exps_%s=%s' % (il, proj, tier, recipe[tier][proj]))
    return rules


def estimate(recipe, layers, counts):
    s = 0.0
    for tier in TIER:
        per = sum(BPW[recipe[tier][p]] for p in PROJ)
        s += counts[tier] * per
    n_expert = sum(counts.values())
    expert_bpw = s / (n_expert * 3)
    expert_mib = len(layers) * s / 8.0
    return expert_bpw, expert_mib, NON_EXPERT_MIB + expert_mib, s


def report(recipe, layers, counts, keep, rules, est, target, tol):
    expert_bpw, expert_mib, total_mib, s = est
    print('=' * 72)
    print('配方 %s   |   层数 %d   |   各档专家数 %s' % (
        args_recipe_label, len(layers), {t: counts[t] for t in TIER}))
    print('=' * 72)
    print('%-5s %-6s %-8s %6s %7s %10s' % ('档', '投影', '类型', 'bpw', '专家数', '体积贡献MiB'))
    for tier in TIER:
        for proj in PROJ:
            t = recipe[tier][proj]
            print('%-5s %-6s %-8s %6.3f %7d %10.1f' % (
                tier, proj, t, BPW[t], counts[tier],
                len(layers) * counts[tier] * BPW[t] / 8.0))
    print('-' * 72)
    print('非专家常数          : %8.0f MiB  (attention/norm/router/shexp/embd/output/blk.40)' % NON_EXPERT_MIB)
    print('专家加权 bpw        : %8.4f' % expert_bpw)
    print('专家部分            : %8.0f MiB' % expert_mib)
    print('预估总体积          : %8.0f MiB = %.2f GiB' % (total_mib, total_mib / 1024.0))
    if target:
        d = total_mib - target
        flag = 'OK' if abs(d) <= tol else 'FAIL'
        print('目标体积            : %8.0f MiB +/- %.0f  ->  差 %+0.0f MiB  %s' % (
            target, tol, d, flag))
        if abs(d) > tol:
            print('')
            print('!! 预估体积偏离目标，中止（不写文件、不量化）')
            sys.exit(2)
    print('-' * 72)
    print('规则: 共 %d 条 = keep %d + 分档 %d (层 %d x 投影 %d x 档 %d)' % (
        len(rules), len(keep), len(rules) - len(keep), len(layers), len(PROJ), len(TIER)))
    from collections import Counter
    types = Counter(r.split('=')[-1] for r in rules)
    print('类型分布: %s' % dict(sorted(types.items(), key=lambda kv: -kv[1])))
    unknown = [t for t in types if t not in BPW]
    if unknown:
        die('规则里有未知类型: %s' % unknown)
    print('-' * 72)
    print('keep 规则（原样继承）:')
    for r in keep:
        print('   ', r)
    print('分档规则样例（前 9 条）:')
    for r in rules[len(keep):len(keep) + 9]:
        print('   ', r)
    print('=' * 72)
    return total_mib


def write_rules(path, rules):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        die('规则输出目录不存在: %s' % d)
    with open(path, 'w') as f:
        f.write('\n'.join(rules) + '\n')
    print('规则已写入: %s (%d 行)' % (path, len(rules)))


def do_quant(args, rules_path, label):
    qbin = args.quant_bin
    if not os.path.exists(qbin):
        die('量化器不存在: %s' % qbin)
    for p, tag in ((args.imatrix, 'imatrix'), (args.split, 'split 模型')):
        if not os.path.exists(p):
            die('%s 不存在: %s' % (tag, p))
    if os.path.exists(args.out):
        die('输出模型已存在，先删或换 --out: %s' % args.out)

    cmd = [qbin, '--imatrix', args.imatrix, '--tensor-type-file', rules_path,
           args.split, args.out, 'q3_K']
    print('')
    print('EXEC: %s' % ' '.join(cmd))
    t0 = time.time()
    with open(args.log, 'w') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=os.path.dirname(qbin) or '.')
        while proc.poll() is None:
            time.sleep(15)
            tail = tail_lines(args.log, 1)
            print('   [%4.0fs] %s' % (time.time() - t0, tail.strip()[:110]))
    txt = open(args.log, errors='replace').read()
    if proc.returncode != 0:
        print(txt[-3000:])
        die('量化失败 rc=%d' % proc.returncode)
    m = re.search(r'quant size\s*=\s*([0-9.]+)\s*MiB', txt)
    if not m:
        print(txt[-3000:])
        die('日志里找不到 quant size')
    actual = float(m.group(1))
    n_missing = txt.count('did not find weights')
    n_bad = txt.count('imatrix size')
    print('')
    print('量化完成  耗时 %.0f 秒' % (time.time() - t0))
    print('  实际 quant size : %.2f MiB' % actual)
    print('  did not find    : %d 条 (期望 13: output/token_embd/blk.40)' % n_missing)
    print('  imatrix size 异常: %d 条 (必须为 0)' % n_bad)
    if n_bad:
        die('出现 imatrix size 不匹配，imatrix 与张量对不上')
    return actual, txt


def do_ppl(args, label):
    if not os.path.exists(args.ppl_bin):
        die('PPL 二进制不存在: %s' % args.ppl_bin)
    if not os.path.exists(args.corpus):
        die('语料不存在: %s' % args.corpus)
    log = args.log.replace('.log', '.ppl.log') if args.log.endswith('.log') else args.log + '.ppl.log'
    cmd = [args.ppl_bin, '-m', args.out, '-f', args.corpus, '-t', str(args.threads)]
    print('')
    print('EXEC: %s' % ' '.join(cmd))
    t0 = time.time()
    with open(log, 'w') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(20)
            tail = tail_lines(log, 1)
            print('   [%4.0fs] %s' % (time.time() - t0, tail.strip()[:110]))
    txt = open(log, errors='replace').read()
    if proc.returncode != 0:
        print(txt[-3000:])
        die('PPL 失败 rc=%d' % proc.returncode)
    m = re.search(r'Final estimate:\s*PPL\s*=\s*([0-9.]+)', txt)
    if not m:
        print(txt[-3000:])
        die('日志里找不到 Final estimate')
    groups = len(re.findall(r'hot', txt, re.I))
    print('')
    print('PPL [%s] = %s   (%.0f 秒, 分档日志命中 %d 次)' % (
        label, m.group(1), time.time() - t0, groups))
    return float(m.group(1))


def tail_lines(path, n):
    try:
        with open(path, errors='replace') as f:
            lines = f.readlines()
        return ''.join(lines[-n:])
    except Exception:
        return ''


def main():
    global args_recipe_label
    a = parse_args()

    if a.list:
        print('内置配方:')
        for name, rec in RECIPES.items():
            flat = '  '.join('%s=%s/%s/%s' % (t, rec[t]['gate'], rec[t]['up'], rec[t]['down'])
                             for t in TIER)
            print('  %-14s %s' % (name, flat))
        print('（每项顺序 = gate/up/down）')
        return

    if a.recipe not in RECIPES:
        die('未知配方 %s（--list 查看）' % a.recipe)
    args_recipe_label = a.recipe
    # derive artefact paths from the recipe name so runs never collide
    if a.out is None:
        a.out = '/root/graded-%s.gguf' % a.recipe
    if a.rules is None:
        a.rules = '/mnt/workspace/moe/ttypes_%s.txt' % a.recipe
    if a.log is None:
        a.log = '/root/quant_%s.log' % a.recipe
    recipe = {t: dict(p) for t, p in RECIPES[a.recipe].items()}
    recipe = apply_overrides(recipe, a.set)

    # 层数 / 各档专家数
    if a.assume:
        layers, counts = parse_assume(a.assume, a.assume_layers)
        print('（使用 --assume，未扫描 split 模型）')
    else:
        layers, counts = scan_split(a.split)
        print('（已扫描 split: %s）' % a.split)

    keep = read_keep(a.template)
    rules = build_rules(recipe, layers, keep)
    est = estimate(recipe, layers, counts)
    total = report(recipe, layers, counts, keep, rules, est, a.target_mib, a.tol)

    if not a.run:
        print('')
        print('dry-run 结束：未写文件、未执行量化。确认无误后加 --run 执行。')
        return

    write_rules(a.rules, rules)
    actual, _qlog = do_quant(a, a.rules, a.recipe)

    # 实测体积 vs 预估
    d = actual - est[2]
    print('  实测 vs 预估    : %+0.0f MiB (%.1f%%)' % (d, 100.0 * d / est[2]))
    if abs(d) > a.tol:
        die('实测体积偏离预估 %+0.0f MiB > %.0f，标定公式或类型表有问题' % (d, a.tol))
    if a.target_mib and abs(actual - a.target_mib) > a.tol:
        die('实测 %.0f MiB 偏离目标 %.0f 超出 %.0f' % (actual, a.target_mib, a.tol))
    print('  体积核对        : OK (%.0f MiB = %.2f GiB)' % (actual, actual / 1024.0))

    if a.ppl:
        do_ppl(a, a.recipe)


if __name__ == '__main__':
    args_recipe_label = '?'
    main()
