"""Generate LLM paraphrases of the clean training instructions for random_aug (instruction.prob).

The randomized setting is evaluated with unseen instructions, so training sees
several rewordings of every instruction. Only the training instructions (meta/tasks.parquet) are
used; RoboTwin's unseen instruction templates are never read. Paraphrases that change an arm,
colour or number word are dropped. Resumable: existing entries in --out are kept.

    # local Qwen3-VL-4B-Instruct (GPU or CPU)
    python scripts/random_aug/gen_instruction_paraphrases.py --device cuda
    # any OpenAI-compatible API (key in OPENAI_API_KEY)
    python scripts/random_aug/gen_instruction_paraphrases.py --backend openai --base_url <url> --model <name>
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common import DEFAULT_DATA, REPO

PROMPT = """Rewrite the robot instruction below in {n} different ways, as a person would naturally say it to a robot.
Rules:
- Same task: same objects, same order of steps, same target places.
- Mention an arm only if the instruction does, and then exactly the same one (left arm / right arm / both arms). Never assign objects to arms.
- Phrases like "with knobs" or "with red lid" describe the object, not how to do the task. Keep colours, numbers and the attributes needed to tell objects apart; other adjectives may be dropped or the object described differently (e.g. "the bottle with red lid" -> "the red-capped bottle").
- Fluent, grammatical English imperative sentences with articles ("the"); no dashes, colons, semicolons or arrows.
- Vary verbs and sentence structure. About a third short (at most 6 words), the rest of normal length.
- Output only a JSON list of {n} strings.
Instruction: "{instruction}\""""

# left / right (arm or direction) must match exactly, and so must "both arms"; a plain "both" / "first"
# / "one after another" may be reworded
SIDE_WORDS = {"left", "right"}
BOTH_ARMS = re.compile(r"\b(both|two)\s+(arms?|hands?|grippers?)\b")
COLOR_WORDS = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "gray", "grey", "brown",
    "silver", "golden", "gold", "beige", "cyan", "violet",
}
NUMBER_WORDS = {"two", "three", "four", "five", "1", "2", "3", "4", "5"}


def _words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower().replace("grey", "gray")))


def validate(original, candidates):
    ow = _words(original)
    out, seen = [], {original.strip().lower()}
    for c in candidates:
        if not isinstance(c, str):
            continue
        c = " ".join(c.strip().strip('"').split())
        cw = _words(c)
        if not (2 <= len(c.split()) <= 45) or c.lower() in seen or re.search(r"[;:–—→]| - ", c):
            continue
        if (cw & SIDE_WORDS) != (ow & SIDE_WORDS):
            continue
        if bool(BOTH_ARMS.search(c.lower())) != bool(BOTH_ARMS.search(original.lower())):
            continue
        if not (ow & COLOR_WORDS) <= cw or not (ow & NUMBER_WORDS) <= cw:
            continue
        seen.add(c.lower())
        out.append(c)
    return out


def parse_list(text):
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


class HFBackend:
    def __init__(self, model_path, device, max_new_tokens):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
        self.device, self.max_new_tokens = device, max_new_tokens

    def __call__(self, prompts):
        texts = [
            self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        batch = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **batch, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.9, top_p=0.95
            )
        return self.tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)


class OpenAIBackend:
    def __init__(self, base_url, model, workers):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY"))
        self.model, self.workers = model, workers

    def _one(self, prompt):
        try:
            r = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.9
            )
            return r.choices[0].message.content or ""
        except Exception as e:  # keep going; the instruction is retried on the next run
            print(f"request failed: {e!r}")
            return ""

    def __call__(self, prompts):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._one, prompts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--out", default=str(REPO / "lingbot-vla-v2/assets/random_aug/instruction_paraphrases.json"))
    p.add_argument("--backend", choices=["hf", "openai"], default="hf")
    p.add_argument("--model", default="/mnt/datadisk/models/lingbot-vla/Qwen3-VL-4B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--base_url", default=None)
    p.add_argument("--num", type=int, default=8, help="paraphrases requested per instruction")
    p.add_argument("--min_keep", type=int, default=3, help="retry instructions with fewer valid paraphrases")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8, help="openai backend: concurrent requests")
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--limit", type=int, default=None, help="only the first N instructions (for a quick test)")
    a = p.parse_args()

    instructions = [s.strip() for s in pd.read_parquet(Path(a.data).expanduser() / "meta/tasks.parquet").index]
    instructions = sorted(set(instructions))[: a.limit]
    out_path = Path(a.out)
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    backend = (
        HFBackend(a.model, a.device, a.max_new_tokens)
        if a.backend == "hf"
        else OpenAIBackend(a.base_url, a.model, a.workers)
    )

    for rnd in range(a.rounds):
        todo = [s for s in instructions if len(result.get(s, [])) < a.min_keep]
        print(f"round {rnd + 1}: {len(todo)} instructions to (re)generate")
        for start in range(0, len(todo), a.batch_size):
            chunk = todo[start:start + a.batch_size]
            replies = backend([PROMPT.format(n=a.num, instruction=s) for s in chunk])
            for s, reply in zip(chunk, replies):
                merged = validate(s, result.get(s, []) + parse_list(reply))
                result[s] = merged[: a.num]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(dict(sorted(result.items())), ensure_ascii=False, indent=1))
            done = sum(len(result.get(s, [])) >= a.min_keep for s in instructions)
            print(f"  {start + len(chunk)}/{len(todo)}  ({done}/{len(instructions)} with >= {a.min_keep})")
    counts = [len(result.get(s, [])) for s in instructions]
    print(f"saved {out_path}: {sum(c > 0 for c in counts)}/{len(instructions)} instructions, "
          f"{sum(counts) / max(1, len(counts)):.1f} paraphrases each on average")


if __name__ == "__main__":
    main()
