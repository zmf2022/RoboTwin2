"""Preview random_aug on random clean training samples (no model needed).

Each sample is two rows: original / augmented; columns: head, left wrist, right wrist (current
frame), head (future frame). Instructions before / after are printed.

    python scripts/random_aug/preview.py --out preview.png --num 6
"""

import argparse
import time

import cv2
import numpy as np
import torch
from torchvision.transforms.v2 import Resize

from common import CAM_KEYS, CleanFrames, DEFAULT_DATA, REPO
from lingbotvla.data.vla_data.random_aug import RandomSceneAugmentor, load_random_aug_config

TRAIN_KEYS = ("observation.images.camera_top", "observation.images.camera_wrist_left", "observation.images.camera_wrist_right")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO / "scripts/random_aug/random_aug.yaml"))
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--out", default="random_aug_preview.png")
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--img_size", type=int, default=256, help="training resize (data.img_size, default 256)")
    p.add_argument("--chunk_size", type=int, default=50)
    p.add_argument("--no_force", action="store_true", help="keep prob as configured instead of 1.0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", default=None, help="comma-separated episode indices to sample from (default: all)")
    a = p.parse_args()

    cfg = load_random_aug_config(a.config)
    if not a.no_force:
        cfg["prob"] = 1.0
    aug = RandomSceneAugmentor(cfg)
    torch.manual_seed(a.seed)
    frames = CleanFrames(a.data)
    rng = np.random.default_rng(a.seed)
    resize = Resize((a.img_size, a.img_size))

    rows, times = [], []
    pool = [int(x) for x in a.episodes.split(",")] if a.episodes else len(frames)
    for ep in rng.choice(pool, a.num, replace=False):
        ep = int(ep)
        n = frames.length(ep)
        t = int(rng.integers(0, n))
        tf = min(n - 1, t + a.chunk_size - 1)
        reqs = [(ep, t, k) for k in CAM_KEYS] + [(ep, tf, k) for k in CAM_KEYS]
        imgs = [resize(torch.from_numpy(x).permute(2, 0, 1)) for x in frames.read(reqs)]
        batch = {
            "image": dict(zip(TRAIN_KEYS, imgs[:3])),
            "future_image": dict(zip(TRAIN_KEYS, imgs[3:])),
            "prompt": [frames.instruction(ep)],
        }
        orig = [x.clone() for x in imgs[:3]] + [imgs[3].clone()]
        start = time.time()
        aug(batch, episode_index=ep, frame_index=t, future_offset=a.chunk_size - 1)
        times.append(time.time() - start)
        new = [batch["image"][k] for k in TRAIN_KEYS] + [batch["future_image"][TRAIN_KEYS[0]]]
        print(f"ep {ep} frame {t}:\n  {frames.instruction(ep)}\n  -> {batch['prompt'][0]}")
        rows.append(np.concatenate([x.permute(1, 2, 0).numpy() for x in orig], 1))
        rows.append(np.concatenate([x.permute(1, 2, 0).numpy() for x in new], 1))
    cv2.imwrite(a.out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f"saved {a.out}; augment {np.mean(times) * 1000:.1f} ms/sample (3 cams x 2 frames)")


if __name__ == "__main__":
    main()
