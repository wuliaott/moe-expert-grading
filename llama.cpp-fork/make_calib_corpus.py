#!/usr/bin/env python3
"""
Build the agentic calibration corpus for Qwen3.6-35B-A3B imatrix.

WHY THIS EXISTS (crucible v2 lesson):
  v1's imatrix flattened chat conversations to plain text - every <|im_start|>,
  role marker and thinking tag was discarded. An imatrix measures activation
  statistics, so v1 measured them on text the model never actually receives.
  v2 fixed this and measurably improved quality at identical file size.

  -> llama-imatrix must run with --parse-special
  -> this corpus MUST contain the real special tokens

Also follows Risa (arXiv 2608.22191): 72% of token surprisal in software-agent
trajectories sits in the lowest-probability quartile - the decision tokens
(diffs, tool-call JSON, assertions). This corpus over-samples those rather than
using uniform long prose.
"""
import json
import os
import random
import sys

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/moe/calib"
os.makedirs(OUT_DIR, exist_ok=True)

IM_S = chr(60) + "|im_start|" + chr(62)
IM_E = chr(60) + "|im_end|" + chr(62)
# NOTE: Qwen tool-call tags carry a zero-width space (U+200B) after the opening
# "<". Written with an explicit escape so the source can never acquire an
# invisible character by accident.
ZWSP = chr(0x200B)
TC_OPEN  = "<" + ZWSP + "tool_call>"
TC_CLOSE = "</" + ZWSP + "tool_call>"
TR_OPEN, TR_CLOSE = "<tool_response>", "</tool_response>"
THINK_OPEN, THINK_CLOSE = chr(60)+"think"+chr(62), chr(60)+"/think"+chr(62)
FUNC_OPEN, FUNC_CLOSE = chr(60)+"function=", chr(60)+"/function"+chr(62)
PARAM_OPEN, PARAM_CLOSE = chr(60)+"parameter=", chr(60)+"/parameter"+chr(62)

# NOTE: Qwen3.6 tool calls are XML-shaped, NOT JSON:
#   <tool_call>
#   <function=NAME>
#   <parameter=ARG>
#   value
#   </parameter>
#   </function>
#   </tool_call>
# Verified against the chat_template in unsloth/Qwen3.6-35B-A3B-GGUF's HF metadata.

SYSTEM = (
    "You are a coding agent working in a repository. You read files, run shell "
    "commands, and produce patches.\n"
    "Available tools:\n"
    "- read_file(path, offset, limit) -> file contents\n"
    "- edit_file(path, old, new) -> edit result\n"
    "- run_cmd(cmd) -> stdout/stderr\n"
    "- search(q) -> matching lines\n"
    "Emit tool calls as a tool_call block. Be precise. Never guess file contents."
)

# ---- decision-token-dense fragments ---------------------------------------
DIFFS = [
    """--- a/src/dates.py
+++ b/src/dates.py
@@ -11,4 +11,3 @@ def parse_iso(s):
     if m.group(2):
         return datetime.strptime(s, "%Y-%m-%dT%H:%M")
-    return datetime.fromisoformat(s)
+    return _parse_offset(s)""",
    """--- a/src/queue.rs
+++ b/src/queue.rs
@@ -44,7 +44,8 @@ impl<T> Queue<T> {
-    fn push(&mut self, v: T) { self.buf[self.tail] = v; self.tail += 1; }
+    fn push(&mut self, v: T) {
+        let t = self.tail;
+        self.buf[t] = v;
+        self.tail = (t + 1) % self.cap;
+    }""",
]

SHELLS = [
    "pytest tests/ -x -vv 2>&1 | tail -40",
    "rg -n 'def parse_iso' --type py src/ tests/",
    "git diff --stat && git stash list",
    "cargo test --release 2>&1 | grep -E 'test result|FAILED|panicked'",
    "python -c \"import json,sys; d=json.load(open(sys.argv[1])); print(list(d)[:20])\" fixtures/a.json",
]

ASSERTS = [
    'assert parse_iso("2026-04-15T10:00:00Z") == datetime(2026, 4, 15, 10, 0, tzinfo=timezone.utc)',
    'assert_queue.push(1); assert_queue.push(2); assert_queue.pop() == 1',
    'assert result.status_code == 200 and result.json()["count"] == 3',
    "assert PPL(quant) - PPL(base) < 0.05, f'kl drift {PPL(quant)-PPL(base):.3f}'",
]

CODE = [
    "```python\ndef parse_iso(s):\n    return datetime.fromisoformat(s)\n```",
    "```rust\nfn split<T: Copy>(v: &[T]) -> (&[T], &[T]) { v.split_at(v.len() / 2) }\n```",
    "```bash\nset -euo pipefail\npytest -q 2>&1 | tail -20\n```",
    "```json\n{\"name\": \"run_cmd\", \"arguments\": {\"cmd\": \"pytest -q\"}}\n```",
]

REASONS = [
    "The assertion compares naive against tz-aware datetimes; the branch attaches "
    "UTC whenever the string carries no offset.",
    "Downstream tests pass tz-suffixed strings, so the explicit strptime branch "
    "shadows the native parser and must go.",
    "tail wraps modulo capacity; without it the tail index grows unbounded and the "
    "ring degenerates to a list.",
    "The mock returns a dict without the count key, so result.json()['count'] raises "
    "KeyError before the status check ever runs.",
]


def strip_tags(text):
    """Remove every special-token tag so we can assert no stray ZWSP remains."""
    for t in (TC_OPEN, TC_CLOSE, TR_OPEN, TR_CLOSE, THINK_OPEN, THINK_CLOSE,
              FUNC_OPEN, FUNC_CLOSE, PARAM_OPEN, PARAM_CLOSE):
        text = text.replace(t, "")
    return text


def tool_call(name, args):
    """Qwen3.6 XML-shaped tool call (verified against the HF chat_template)."""
    parts = [TC_OPEN, "\n", FUNC_OPEN, name, ">\n"]
    for k, v in args.items():
        if isinstance(v, dict):
            body = json.dumps(v, indent=2)
        elif isinstance(v, list):
            body = "\n".join(str(x) for x in v)
        else:
            body = str(v)
        parts += [PARAM_OPEN, k, ">\n", body, "\n", PARAM_CLOSE, "\n"]
    parts += [FUNC_CLOSE, "\n", TC_CLOSE, "\n"]
    return "".join(parts)


def build_dialogue(rng):
    """One realistic agent trajectory: think -> call -> observe -> correct."""
    parts = [f"{IM_S}system\n{SYSTEM}{IM_E}"]

    n_turns = rng.randint(2, 5)
    for turn in range(n_turns):
        goal = rng.choice([
            "The test suite fails on test_parse_iso. Run it and show the output.",
            "search() returns nothing for 'parse_iso'. Find where it is defined.",
            "Add a bounded lock-free queue in Rust and test it.",
            "The p95 latency regressed after the last commit. Bisect it.",
            "run_cmd returns exit 127. Work out why and fix it.",
        ])
        parts.append(f"{IM_S}user\n{goal}{IM_E}")
        parts.append(f"{IM_S}assistant\n")

        # A3B is a heavy thinker and thinking text is most of the token volume.
        # Omitting it would profile a token stream the model never actually sees.
        if rng.random() < 0.7:
            parts.append(THINK_OPEN + "\n")
            for _ in range(rng.randint(2, 5)):
                parts.append(rng.choice(REASONS) + " ")
            parts.append("\n" + THINK_CLOSE + "\n\n")

        for _ in range(rng.randint(1, 3)):
            parts.append(rng.choice(REASONS) + "\n")
            parts.append(tool_call("run_cmd", {"cmd": rng.choice(SHELLS)}))
            parts.append(
                f"{TR_OPEN}\n{rng.choice(ASSERTS)}\n"
                f"E       AssertionError\n"
                f"1 failed, {rng.randint(5, 40)} passed in {rng.uniform(.2, 9):.2f}s\n{TR_CLOSE}"
            )
            parts.append(tool_call("read_file", {"path": rng.choice(
                ["src/dates.py", "src/queue.rs", "tests/test_dates.py", "src/api.py"])}))
            parts.append(
                f"{TR_OPEN}\n{rng.choice(CODE)}\n{TR_CLOSE}"
            )

        if rng.random() < 0.6:
            parts.append("Applying the patch:\n" + rng.choice(DIFFS))

        parts.append(rng.choice(CODE))
        parts.append(f"{IM_E}")

    return "".join(parts)


def main():
    rng = random.Random(42)  # crucible also used seed=42; keep it comparable
    n_samples = int(os.environ.get("N_SAMPLES", "512"))

    dialogs = [build_dialogue(rng) for _ in range(n_samples)]

    # Deduplicate: identical imatrix windows contribute zero new information
    # but still cost forward passes.
    seen, uniq = set(), []
    for d in dialogs:
        h = hash(d)
        if h not in seen:
            seen.add(h)
            uniq.append(d)

    path = os.path.join(OUT_DIR, "agent_calib.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(uniq))

    total_chars = sum(len(d) for d in uniq)
    print(f"samples requested : {n_samples}")
    print(f"samples written   : {len(uniq)}  ({len(uniq)/n_samples:.1%} unique)")
    print(f"path              : {path}")
    print(f"total chars       : {total_chars:,}  (~{total_chars//4:,} tokens)")
    print()
    print("Sanity checks (must all be TRUE):")
    ok = True
    blob = "".join(uniq)
    for name, cond in [
        ("contains <|im_start|>", IM_S in blob),
        ("contains <|im_end|>", IM_E in blob),
        ("contains tool_call tag", TC_OPEN in blob and TC_CLOSE in blob),
        ("contains <function=>", FUNC_OPEN in blob and FUNC_CLOSE in blob),
        ("contains <parameter=>", PARAM_OPEN in blob and PARAM_CLOSE in blob),
        ("contains <tool_response>", TR_OPEN in blob and TR_CLOSE in blob),
        ("contains <think>", THINK_OPEN in blob and THINK_CLOSE in blob),
        ("contains unified diffs", "@@ -" in blob),
        ("contains shell pipes", "| tail" in blob),
        # Qwen tool-call tags legitimately contain U+200B. What we must forbid
        # is a zero-width char OUTSIDE the tag delimiters (an invisible typo).
        ("exactly one ZWSP per tool_call tag",
         blob.count(ZWSP) == blob.count(TC_OPEN) + blob.count(TC_CLOSE)),
        ("no stray ZWSP outside tags", ZWSP not in strip_tags(blob)),
    ]:
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        ok &= cond
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()