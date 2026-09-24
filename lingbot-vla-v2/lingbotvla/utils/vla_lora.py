"""In-place LoRA for the Qwen-VL LLM inside LingBot-VLA (no peft dependency).

Base weights keep their original names (`<linear>.weight`), LoRA adds `<linear>.lora_A`,
`<linear>.lora_B` and a `<linear>.lora_scale` buffer, so training checkpoints resume
as usual and `merge_lora_state_dict` folds the adapters back into a plain HF state dict.
"""
import math
import re
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_SUFFIXES = (".lora_A", ".lora_B", ".lora_scale")


class LoRALinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        return out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scale


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    target_modules: Sequence[str],
    scope: str = "qwenvl_with_expert.qwenvl.model.language_model.",
    frozen_scope: str = "qwenvl_with_expert.qwenvl.",
) -> int:
    """Freeze everything under `frozen_scope`, then add LoRA to Linear layers under `scope`
    whose last name component is in `target_modules`. Returns the number of LoRA params."""
    for name, p in model.named_parameters():
        if frozen_scope in name:
            p.requires_grad_(False)

    pattern = re.compile(r"\.(%s)$" % "|".join(map(re.escape, target_modules)))
    n_params = 0
    for name, module in model.named_modules():
        if scope not in name or type(module) is not nn.Linear or not pattern.search(name):
            continue
        w = module.weight
        module.lora_A = nn.Parameter(torch.empty(rank, module.in_features, device=w.device, dtype=w.dtype))
        module.lora_B = nn.Parameter(torch.zeros(module.out_features, rank, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
        module.register_buffer("lora_scale", torch.tensor(alpha / rank, device=w.device, dtype=w.dtype))
        module.__class__ = LoRALinear
        n_params += module.lora_A.numel() + module.lora_B.numel()
    if n_params == 0:
        raise ValueError(f"LoRA matched no Linear under '{scope}' with targets {list(target_modules)}")
    return n_params


def merge_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Fold `W += scale * B @ A` and drop LoRA keys. No-op for state dicts without LoRA."""
    for key_a in [k for k in state_dict if k.endswith(".lora_A")]:
        prefix = key_a[: -len(".lora_A")]
        a = state_dict.pop(key_a).float()
        b = state_dict.pop(prefix + ".lora_B").float()
        scale = state_dict.pop(prefix + ".lora_scale").float()
        w = state_dict[prefix + ".weight"]
        state_dict[prefix + ".weight"] = (w.float() + scale * (b @ a)).to(w.dtype)
    return state_dict
