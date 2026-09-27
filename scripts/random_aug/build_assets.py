"""Build the random_aug assets from the clean training set only.

1. Head camera empty-scene reference: per-pixel median of background-like pixels over random
   frames of all episodes (arms and objects move between frames, the table / wall / static
   shadows do not).
2. Distractor library: objects segmented from the first and last head frame of every episode
   (components not touching the image border nor the rendered arm masks, if render_arm_masks.py
   has been run), stored as RGBA cut-outs with
   their task id, plus a task conflict matrix (tasks whose instructions name the same kind of
   object, e.g. all block tasks) so a sample never gets distractors that its instruction could
   refer to.

    python scripts/random_aug/build_assets.py
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from common import DEFAULT_DATA, HEAD_KEY, REPO, CleanFrames
from lingbotvla.data.vla_data.random_aug import DEFAULTS, ArmMasks, HeadReference, compute_background


# object nouns of the RoboTwin tasks (from the clean instructions), synonyms mapped together
OBJECT_NOUNS = {
    "bottle", "bell", "rack", "mug", "cup", "stapler", "mat", "pad", "box", "plasticbox", "toycar", "basket",
    "can", "fan", "phone", "phonestand", "paymentsign", "block", "hammer", "tabletrashbin", "bin", "dustbin",
    "balls", "kitchenpot", "pot", "laptop", "mouse", "shoe", "scanner", "bowl", "roller", "sauce", "lid",
    "microwave", "bread", "breadbasket", "loaf", "container", "plate", "playingcards", "cards", "skillet",
    "electronicscale", "scale", "seal", "clock", "microphone", "hamburg", "tray", "fries", "coaster",
    "displaystand", "cabinet", "drawer", "switch",
}
SYNONYMS = {
    "mug": "cup", "tabletrashbin": "bin", "dustbin": "bin", "plasticbox": "box", "kitchenpot": "pot",
    "loaf": "bread", "cards": "playingcards", "electronicscale": "scale", "pad": "mat",
}


def task_conflicts(frames, episodes_per_task, min_count=5):
    """[T, T] bool: tasks whose instructions share an object noun (always True on the diagonal)."""
    words = defaultdict(Counter)
    for ep in frames.episodes.index:
        task = ep // episodes_per_task
        words[task].update(set(re.findall(r"[a-z]+", frames.instruction(ep).lower())))
    num = max(words) + 1
    nouns = [{SYNONYMS.get(w, w) for w, n in words[t].items() if n >= min_count and w in OBJECT_NOUNS} for t in range(num)]
    conflict = np.eye(num, dtype=bool)
    for i in range(num):
        for j in range(num):
            conflict[i, j] |= bool(nouns[i] & nouns[j])
    return conflict


def build_reference(frames, num, rng):
    eps = rng.integers(0, len(frames), num)
    reqs = [(int(e), int(rng.integers(0, frames.video_length(int(e)))), HEAD_KEY) for e in eps]
    imgs = np.stack(frames.read(reqs)).astype(np.float32) / 255.0  # N,H,W,3
    mx, mn = imgs.max(-1), imgs.min(-1)
    # background-like: bright and unsaturated (table, wall, light shadows); excludes the black
    # arm parts and coloured objects
    valid = ((mx - mn) / (mx + 1e-6) < 0.12) & (imgs.mean(-1) > 0.55)
    imgs[~valid] = np.nan
    ref = np.nanmedian(imgs, axis=0)
    missing = valid.sum(0) < max(10, num // 50)
    ref = (np.nan_to_num(ref, nan=0.0) * 255).round().astype(np.uint8)
    if missing.any():
        ref = cv2.inpaint(ref, missing.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    print(f"reference: {num} frames, {missing.mean() * 100:.2f}% pixels inpainted")
    return ref


def harvest(img, mcfg, ref, arm=None):
    """Object cut-outs (RGBA uint8, center row) from one head frame; ``arm``: bool robot mask or None."""
    table_top = ref["table_top"]
    m = compute_background(img.astype(np.float32) / 255.0, mcfg, ref, arm)
    labels, num = ndimage.label(~m["bg"])
    h, w = labels.shape
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    lum = img.mean(-1) / 255.0
    ratio = lum / (ref["lum"] + 1e-6)
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None or i in border:
            continue
        comp = labels[sl] == i
        if arm is not None and (comp & arm[sl]).mean() > 0.02:
            continue
        area = int(comp.sum())
        ch, cw = comp.shape
        # thin slivers near the far table edge are wall / table seam artefacts
        if not (60 <= area <= 6000) or ch < 8 or cw < 6 or sl[0].start < table_top + 8:
            continue
        # pure shadow: only slightly darker than the table and uncoloured
        rgb = img[sl][comp].astype(np.float32) / 255.0
        sat = ((rgb.max(-1) - rgb.min(-1)) / (rgb.max(-1) + 1e-6)).mean()
        if sat < 0.08 and ratio[sl][comp].mean() > 0.6:
            continue
        alpha = cv2.erode(comp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)
        rgba = np.dstack([img[sl], (alpha * 255).round().astype(np.uint8)])
        out.append((rgba, (sl[0].start + sl[0].stop) / 2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--ref_out", default=str(REPO / "lingbot-vla-v2/assets/random_aug/cam_high_ref.png"))
    p.add_argument("--distractor_out", default=str(REPO / "data/random_aug/distractors.npz"))
    p.add_argument("--arm_masks", default=str(REPO / "data/random_aug/arm_masks.npy"),
                   help="render_arm_masks.py output; skipped if missing")
    p.add_argument("--ref_frames", type=int, default=600)
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--preview", default=None, help="save a contact sheet of the cut-outs here")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    frames = CleanFrames(a.data)
    ref_img = build_reference(frames, a.ref_frames, rng)
    Path(a.ref_out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.ref_out, cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
    print(f"saved {a.ref_out}")

    mcfg = DEFAULTS["mask"]
    ref = HeadReference(ref_img, mcfg).at(*ref_img.shape[:2])
    arms = ArmMasks(a.arm_masks) if Path(a.arm_masks).exists() else None
    print(f"arm masks: {a.arm_masks if arms else 'none (colour rules only)'}")
    h, w = ref_img.shape[:2]
    crops, center_y, task = [], [], []
    eps = list(frames.episodes.index)
    for start in range(0, len(eps), 100):
        batch = eps[start:start + 100]
        reqs = [(ep, f, HEAD_KEY) for ep in batch for f in (0, frames.video_length(ep) - 1)]
        for (ep, f, _), img in zip(reqs, frames.read(reqs)):
            arm = arms.get(ep, f, 0, h, w, dilate_px=2) if arms else None
            for rgba, cy in harvest(img, mcfg, ref, arm):
                crops.append(rgba)
                center_y.append(cy)
                task.append(ep // a.episodes_per_task)
        print(f"episodes {start + len(batch)}/{len(eps)}: {len(crops)} cut-outs")

    conflict = task_conflicts(frames, a.episodes_per_task)
    print(f"task conflicts: {int(conflict.sum() - len(conflict))} ordered pairs besides the task itself")
    shapes = np.array([c.shape[:2] for c in crops], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum([c.size for c in crops])[:-1]]).astype(np.int64)
    Path(a.distractor_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.distractor_out,
        pixels=np.concatenate([c.reshape(-1) for c in crops]),
        offsets=offsets,
        shapes=shapes,
        center_y=np.array(center_y, dtype=np.float32),
        task=np.array(task, dtype=np.int32),
        native_hw=np.array(ref_img.shape[:2], dtype=np.int32),
        conflict=conflict,
    )
    print(f"saved {a.distractor_out}: {len(crops)} cut-outs from {len(set(task))} tasks")

    if a.preview:
        tile, cols = 64, 24
        idx = rng.permutation(len(crops))[: cols * 12]
        sheet = np.full((tile * ((len(idx) + cols - 1) // cols), tile * cols, 3), 255, np.uint8)
        for k, i in enumerate(idx):
            rgba = crops[i]
            s = tile / max(rgba.shape[:2])
            r = cv2.resize(rgba, (max(1, int(rgba.shape[1] * s)), max(1, int(rgba.shape[0] * s))))
            al = r[..., 3:4] / 255.0
            y, x = (k // cols) * tile, (k % cols) * tile
            cell = sheet[y:y + r.shape[0], x:x + r.shape[1]]
            cell[:] = (cell * (1 - al) + r[..., :3] * al).astype(np.uint8)
        Path(a.preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.preview, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"saved {a.preview}")


if __name__ == "__main__":
    main()
