"""Clean -> randomized scene augmentation for clean-only RoboTwin training.

Enabled with ``--data.random_aug_config <yaml>`` (see ``scripts/random_aug/random_aug.yaml``) and applied per
sample in ``FeatureTransform.apply``, before the Qwen-VL image processor. From clean frames only it
simulates the demo_randomized setting:

- background: textures on the table and the wall, keeping the original shading (shadows); the
  robot arms are protected by per-frame masks rendered from the recorded joint states;
- distractor: object cut-outs from clean frames of *other* tasks pasted onto free table area
  (head camera, optionally enlarged on the wrist cameras), occluded by the arm / task objects;
- lighting: global gain, colour temperature, gamma, spatial illumination, rare extreme light;
- geometry: small scale / shift of the head camera (table height +-3cm);
- instruction: LLM paraphrases of the training instruction.

All parameters are sampled once per sample and shared by the current and future frame, so the
future-frame depth / video targets stay consistent with the augmented current frame. (Giving the
depth / video teachers the clean frames instead made the student erase the pasted distractors from
its features, i.e. learn "pasted or not"; real clutter is not pasted and was then taken for the task
objects: stage 2 grasped the clutter in randomized scenes.)
"""

import copy
import json
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import yaml
from scipy import ndimage

from ...utils import logging


logger = logging.get_logger(__name__)


DEFAULTS = {
    # probability that a sample gets the image randomization at all (the rest stays clean)
    "prob": 0.5,
    # substrings of the image keys identifying the fixed head camera / the wrist cameras
    "head_keys": ["camera_top", "cam_high"],
    "wrist_keys": ["wrist"],
    # RoboTwin clean data stores 50 consecutive episodes per task; used to keep distractors from
    # the sample's own task and tasks with the same kind of object
    "episodes_per_task": 50,
    "mask": {
        "head_ref": None,  # empty-scene head camera image built by scripts/random_aug/build_assets.py
        # per-frame robot arm masks (head, left wrist, right wrist) rendered by
        # scripts/random_aug/render_arm_masks.py; the white arm links look like the white table to
        # the colour rules. None = colour rules + head_fill_borders only
        "arm_masks": None,
        "arm_dilate_px": 1,
        "chroma_tol": 0.09,  # head: max chroma difference to the reference for background
        "shadow_min": 0.72,  # head: min luminance ratio to the reference (darker = object / shadow)
        "bright_max": 1.12,  # head: max luminance ratio to the reference
        "sat_tol": 0.15,  # wrist: max saturation for background
        "bright_min": 0.60,  # wrist: min luminance for background
        "edge_tol": 0.10,  # Sobel magnitude (above the reference's) that marks an object outline
        "pink_tol": 0.028,  # R - G above this marks the (pinkish) wall instead of the white table
        "close_px": 2,
        "dilate_px": 1,
        # head, without arm_masks: the arms enter from these image borders; treating them as
        # foreground when filling holes keeps some white arm links from being textured
        "head_fill_borders": ["left", "right", "bottom"],
        "min_bg_blob": 30,  # background islands smaller than this (px at 240x320) become foreground
        # soft shadows / colour bleeding next to objects: pixels this close to the background that
        # are unsaturated, not too dark and not an edge go back to the background (shade keeps them darker)
        "halo_px": 4,
        "halo_sat": 0.3,
        "halo_lum": 0.45,  # head: luminance ratio to the reference; wrist: absolute luminance
        # white table pixels left on object / arm outlines (edge pixels, dilation) show up as a white
        # halo on dark textures, which real randomized frames never have: near-white, unsaturated pixels
        # within rim_px of the background go to it as well. 0 = off
        "rim_px": 0,
        "rim_sat": 0.12,
        "rim_lum": 0.8,  # head: luminance ratio to the reference; wrist: absolute luminance
    },
    "background": {
        "prob": 0.9,  # per randomized sample, for the head camera
        "wrist_prob": 0.8,  # per wrist camera, given the head camera got a texture
        "texture_dir": None,  # directory of texture images (e.g. DTD); None = procedural only
        "texture_dir_prob": 0.7,
        "same_texture_prob": 0.2,  # table and wall share one texture
        "perspective": [1.2, 2.2],  # head table texture: far/near width ratio
        # [lo, hi]: rescale each texture to a mean luminance drawn from this range (randomized eval
        # scenes are dark: head frames average 0.33 vs 0.92 clean); None = texture as loaded
        "brightness": None,
        # [lo, hi]: Gaussian blur sigma (px at the working resolution) on each texture; rendered
        # textures are smoother than DTD photos (Laplacian variance 278 vs 607 on head frames). None = off
        "blur": None,
    },
    "distractor": {
        "prob": 0.7,
        "library": None,  # built by scripts/random_aug/build_assets.py
        "num": [1, 4],
        "clearance_px": 8,  # min gap to the arm / task objects (px at 240x320)
        "scale": [0.8, 1.25],
        "horizon": -0.8,  # vanishing row of the table plane, as a fraction of the image height
        "shadow": 0.8,  # contact shadow darkening (1 = none)
        "top_margin": 0.1,  # head: object centres at least this far (fraction of h) below the table's far edge
        # wrist cameras (per camera, given the sample got distractors); cut-outs are head-camera sized,
        # so they are enlarged by wrist_scale. 0 = head camera only
        "wrist_prob": 0.0,
        "wrist_num": [1, 3],
        "wrist_scale": [1.5, 3.5],
        # pick cut-outs uniformly over their source tasks instead of over cut-outs (the block tasks
        # alone give a quarter of the library; the eval clutter is household objects)
        "task_balance": False,
    },
    "lighting": {
        "prob": 0.8,
        "gain": [0.8, 1.2],
        "temperature": 0.15,  # +-R/B gain for warm / cold light
        "tint": 0.04,
        "gamma": [0.8, 1.25],
        "spatial": 0.3,  # max illumination variation across the image
        "crazy_prob": 0.03,  # extreme coloured / dark / bright light
    },
    "geometry": {
        "prob": 0.5,
        "scale": 0.04,
        "shift": 0.02,  # fraction of the image size
    },
    "instruction": {
        "prob": 0.5,
        "paraphrase_file": None,  # {instruction: [paraphrases]}
        "format_noise": 0.2,  # drop the final period / lowercase the first letter
    },
}

_NATIVE_HW = (240, 320)
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_random_aug_config(path):
    """Read the YAML config; relative asset paths are resolved against the YAML's directory."""
    with open(path) as f:
        cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    root = os.path.dirname(os.path.abspath(path))
    for section, key in (
        ("mask", "head_ref"),
        ("mask", "arm_masks"),
        ("background", "texture_dir"),
        ("distractor", "library"),
        ("instruction", "paraphrase_file"),
    ):
        value = cfg[section].get(key)
        if value:
            cfg[section][key] = os.path.normpath(os.path.join(root, os.path.expanduser(value)))
    return cfg


# ----------------------------------------------------------------------------------------------
# background mask


def _gradient(lum):
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy) / 4.0


class HeadReference:
    """Empty-scene head camera image and derived maps, resized to the working resolution."""

    def __init__(self, ref_rgb, mcfg):
        self.ref_native = ref_rgb.astype(np.float32) / 255.0
        self.mcfg = mcfg
        self._cache = {}

    def at(self, h, w):
        if (h, w) not in self._cache:
            ref = cv2.resize(self.ref_native, (w, h), interpolation=cv2.INTER_AREA)
            lum = ref.mean(-1)
            wall = (ref[..., 0] - ref[..., 1]) > self.mcfg["pink_tol"]
            # the wall is the band above the table edge: keep only rows that are mostly wall
            wall &= (wall.mean(1) > 0.5)[:, None]
            table = ~wall
            self._cache[(h, w)] = {
                "ref": ref,
                "lum": lum,
                "chroma": ref / (lum[..., None] + 1e-6),
                "grad": _gradient(lum),
                "wall": wall,
                "table_white": float(np.percentile(lum[table], 95)) if table.any() else 1.0,
                "wall_white": float(np.percentile(lum[wall], 95)) if wall.any() else 1.0,
                "table_top": int(np.argmax(table.mean(1) > 0.5)),
            }
        return self._cache[(h, w)]


def compute_background(img, mcfg, ref=None, protect=None):
    """Split a clean RoboTwin frame into background (table / wall) and foreground.

    Args:
        img: float32 HxWx3 in [0, 1].
        ref: ``HeadReference.at(h, w)`` for the fixed head camera, None for wrist cameras.
        protect: bool HxW pixels that are always foreground (rendered robot arm), or None.

    Returns dict with ``bg`` (bool, cleaned), ``alpha`` (feathered bg), ``shade`` (luminance of the
    clean background relative to plain white, to keep shadows on the new texture) and ``wall``.
    """
    h, w = img.shape[:2]
    lum = img.mean(-1)
    grad = _gradient(lum)
    mx, mn = img.max(-1), img.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    if ref is not None:
        ratio = lum / (ref["lum"] + 1e-6)
        chroma = img / (lum[..., None] + 1e-6)
        dchroma = np.abs(chroma - ref["chroma"]).max(-1)
        bg = (dchroma < mcfg["chroma_tol"]) & (ratio > mcfg["shadow_min"]) & (ratio < mcfg["bright_max"])
        edge = (grad - ref["grad"]) > mcfg["edge_tol"]
        wall = ref["wall"]
        soft = (sat < mcfg["halo_sat"]) & (ratio > mcfg["halo_lum"])
    else:
        bg = (sat < mcfg["sat_tol"]) & (lum > mcfg["bright_min"])
        edge = grad > mcfg["edge_tol"]
        wall = ((img[..., 0] - img[..., 1]) > mcfg["pink_tol"]).astype(np.uint8) * 255
        # wall / table differ only slightly in tint: smooth away speckles along the seam
        wall = cv2.medianBlur(wall, 2 * max(1, int(round(4 * h / _NATIVE_HW[0]))) + 1) > 127
        soft = (sat < mcfg["halo_sat"]) & (lum > mcfg["halo_lum"])

    scale = h / _NATIVE_HW[0]
    fg = ~bg | edge
    close_px = max(1, int(round(mcfg["close_px"] * scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    borders = mcfg["head_fill_borders"] if (ref is not None and protect is None) else []
    if borders:
        padded = np.pad(fg, 1, constant_values=False)
        for side, sl in (("left", np.s_[:, 0]), ("right", np.s_[:, -1]), ("top", np.s_[0, :]), ("bottom", np.s_[-1, :])):
            if side in borders:
                padded[sl] = True
        fg = ndimage.binary_fill_holes(padded)[1:-1, 1:-1]
    else:
        fg = ndimage.binary_fill_holes(fg)
    # small background islands (inside object outlines that did not close) are foreground
    labels, num = ndimage.label(~fg)
    if num > 0:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, num + 1))
        small = np.zeros(num + 1, dtype=bool)
        small[1:] = sizes < mcfg["min_bg_blob"] * scale * scale
        fg |= small[labels]
    dilate_px = int(round(mcfg["dilate_px"] * scale))
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        fg = cv2.dilate(fg.astype(np.uint8), kernel).astype(bool)
    # closing / dilation, soft shadows and colour bleeding leave a rim of table around every object:
    # give it back to the background (object interiors are further away, saturated or edges)
    halo_px = int(round(mcfg["halo_px"] * scale))
    if halo_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo_px + 1, 2 * halo_px + 1))
        near = cv2.dilate((~fg).astype(np.uint8), kernel).astype(bool)
        fg &= ~(near & (bg | soft) & ~edge)
    rim_px = int(round(mcfg["rim_px"] * scale))
    if rim_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_px + 1, 2 * rim_px + 1))
        near = cv2.dilate((~fg).astype(np.uint8), kernel).astype(bool)
        white = (sat < mcfg["rim_sat"]) & ((lum / (ref["lum"] + 1e-6) if ref is not None else lum) > mcfg["rim_lum"])
        fg &= ~(near & white)
    if protect is not None:
        fg |= protect
    bg = ~fg

    if ref is not None:
        white = np.where(wall, ref["wall_white"], ref["table_white"])
    else:
        white = np.ones_like(lum)
        for region in (wall, ~wall):
            sel = bg & region
            if sel.sum() > 50:
                white[region] = np.percentile(lum[sel], 90)
    shade = np.clip(lum / (white + 1e-6), 0.3, 1.1)
    alpha = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 0.7 * max(scale, 1.0))
    return {"bg": bg, "alpha": alpha, "shade": shade, "wall": wall}


# ----------------------------------------------------------------------------------------------
# textures


def _random_color(rng, near=None):
    """Random colour, biased to muted tones; ``near`` = HSV of a colour to stay close to."""
    if near is None:
        hsv = [rng.integers(0, 180), 255 * rng.random() ** 1.5, rng.integers(50, 256)]
    else:
        hsv = [(near[0] + rng.integers(-12, 13)) % 180, near[1] * rng.uniform(0.6, 1.2), near[2] * rng.uniform(0.5, 1.3)]
    hsv = np.clip(np.array(hsv, np.float32), 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv[None, None], cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32) / 255.0
    return rgb, hsv


def _fractal_noise(rng, size, octaves=4):
    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    cells = int(rng.integers(2, 6))
    for _ in range(octaves):
        grid = rng.random((cells, cells)).astype(np.float32)
        out += amp * cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
        cells *= 2
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def procedural_texture(rng, size=384):
    """Random table / wall texture: solid, noise, stripes, checker or wood-like, float32 [0, 1]."""
    c1, hsv1 = _random_color(rng)
    c2, _ = _random_color(rng, near=hsv1 if rng.random() < 0.6 else None)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    theta = rng.random() * np.pi
    u = xx * np.cos(theta) + yy * np.sin(theta)
    kind = int(rng.integers(0, 5))
    if kind == 0:  # solid with slight noise
        t = 0.15 * _fractal_noise(rng, size)
    elif kind == 1:  # cloudy / marble
        t = _fractal_noise(rng, size, octaves=int(rng.integers(3, 6)))
    elif kind == 2:  # stripes
        period = rng.uniform(8, 64)
        t = 0.5 + 0.5 * np.sin(2 * np.pi * u / period)
        if rng.random() < 0.5:
            t = (t > 0.5).astype(np.float32)
    elif kind == 3:  # checker / tiles
        period = rng.uniform(12, 64)
        v = -xx * np.sin(theta) + yy * np.cos(theta)
        t = ((np.floor(u / period) + np.floor(v / period)) % 2).astype(np.float32)
    else:  # wood grain
        warp = _fractal_noise(rng, size) * rng.uniform(10, 40)
        t = 0.5 + 0.5 * np.sin(2 * np.pi * (u + warp) / rng.uniform(6, 24))
        t = t ** rng.uniform(0.5, 2.0)
    tex = c1 * (1 - t[..., None]) + c2 * t[..., None]
    tex += rng.normal(0, rng.uniform(0, 0.03), tex.shape).astype(np.float32)
    return np.clip(tex, 0, 1)


class TextureSource:
    def __init__(self, bcfg):
        self.bcfg = bcfg
        self.files = []
        tdir = bcfg.get("texture_dir")
        if tdir:
            for root, _, names in os.walk(tdir):
                self.files += [os.path.join(root, n) for n in names if n.lower().endswith(_IMAGE_EXTS)]
            self.files.sort()
            if not self.files:
                logger.warning(f"random_aug: no texture images under {tdir}, using procedural textures only")
        self._cache = OrderedDict()

    def _load(self, path):
        if path not in self._cache:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self._cache[path] = img
            if len(self._cache) > 64:
                self._cache.popitem(last=False)
        return self._cache[path]

    def sample(self, rng, h, w):
        """A texture of at least (h, w), randomly scaled / flipped / colour shifted."""
        tex = None
        if self.files and rng.random() < self.bcfg["texture_dir_prob"]:
            tex = self._load(self.files[int(rng.integers(len(self.files)))])
        if tex is None:
            tex = procedural_texture(rng)
        scale = max(h / tex.shape[0], w / tex.shape[1]) * rng.uniform(1.0, 2.0)
        tex = cv2.resize(tex, (max(w, int(tex.shape[1] * scale)), max(h, int(tex.shape[0] * scale))))
        y0 = int(rng.integers(0, tex.shape[0] - h + 1))
        x0 = int(rng.integers(0, tex.shape[1] - w + 1))
        tex = tex[y0:y0 + h, x0:x0 + w]
        if rng.random() < 0.5:
            tex = tex[:, ::-1]
        if rng.random() < 0.3:
            tex = tex[..., rng.permutation(3)]
        if self.bcfg.get("brightness"):
            tex = np.clip(tex * (rng.uniform(*self.bcfg["brightness"]) / max(float(tex.mean()), 1e-3)), 0, 1)
        if self.bcfg.get("blur"):
            sigma = rng.uniform(*self.bcfg["blur"])
            if sigma > 0.3:
                tex = cv2.GaussianBlur(tex, (0, 0), sigma)
        return np.ascontiguousarray(tex, dtype=np.float32)

    def sample_table(self, rng, h, w, perspective):
        """Table texture with perspective foreshortening for the head camera."""
        tex = self.sample(rng, 2 * h, 2 * w)
        th, tw = tex.shape[:2]
        ratio = rng.uniform(*perspective)
        bottom = tw * rng.uniform(0.35, 0.5)
        top = min(tw, bottom * ratio)
        cx = tw / 2
        src = np.float32([[cx - top / 2, 0], [cx + top / 2, 0], [cx + bottom / 2, th * 0.6], [cx - bottom / 2, th * 0.6]])
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        return cv2.warpPerspective(tex, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REFLECT)


# ----------------------------------------------------------------------------------------------
# distractors


class DistractorLibrary:
    def __init__(self, path):
        data = np.load(path)
        self.pixels = data["pixels"]
        self.offsets = data["offsets"]
        self.shapes = data["shapes"]
        self.center_y = data["center_y"]
        self.task = data["task"]
        self.native_hw = tuple(int(x) for x in data["native_hw"])
        # conflict[i, j]: task j's objects could be what task i's instruction refers to
        self.conflict = data["conflict"] if "conflict" in data.files else None
        _, inverse, counts = np.unique(self.task, return_inverse=True, return_counts=True)
        self.task_balanced_cdf = np.cumsum(1.0 / counts[inverse])
        self.task_balanced_cdf /= self.task_balanced_cdf[-1]

    def sample(self, rng, task_balance=False):
        if not task_balance:
            return int(rng.integers(len(self.shapes)))
        return min(int(np.searchsorted(self.task_balanced_cdf, rng.random(), side="right")), len(self.shapes) - 1)

    def allowed(self, i, own_task):
        if own_task is None:
            return True
        task = int(self.task[i])
        if self.conflict is not None and own_task < len(self.conflict) and task < len(self.conflict):
            return not self.conflict[own_task, task]
        return task != own_task

    def __len__(self):
        return len(self.shapes)

    def get(self, i):
        h, w = self.shapes[i]
        rgba = self.pixels[self.offsets[i]:self.offsets[i] + h * w * 4].reshape(h, w, 4)
        return rgba.astype(np.float32) / 255.0


# ----------------------------------------------------------------------------------------------
# arm masks


class ArmMasks:
    """Bit-packed arm masks [num_frames, cams, h*w/8] + ``<name>_index.npz`` (episode offsets)."""

    def __init__(self, path):
        self.data = np.load(path, mmap_mode="r")
        index = np.load(os.path.splitext(path)[0] + "_index.npz")
        self.ep_from, self.ep_len = index["ep_from"], index["ep_len"]
        # frames each episode's video segment really has (== ep_len on the fixed data; the official
        # unfixed v3.0 misses 1-2 frames, and the rows beyond it decode as the next episode's first frames)
        self.video_len = index["video_len"] if "video_len" in index.files else None
        self.next_same_file = index["next_same_file"] if "next_same_file" in index.files else None
        self.hw = tuple(int(x) for x in index["mask_hw"])
        self.cams = [str(c) for c in index["cams"]]

    def get(self, episode, frame, cam, h, w, dilate_px):
        """bool (h, w) mask of the robot in camera ``cam`` (0 head, 1 left, 2 right), or None."""
        if episode is None or frame is None or not 0 <= episode < len(self.ep_from):
            return None
        frame = min(max(int(frame), 0), int(self.ep_len[episode]) - 1)
        rows = [frame]
        if self.video_len is not None and self.video_len[episode, cam] < self.ep_len[episode]:
            if frame >= self.video_len[episode, cam]:
                if not self.next_same_file[episode, cam]:
                    return None
                frame -= int(self.video_len[episode, cam])
                episode += 1
                rows = [frame]
            else:
                # the missing frames sit mid-episode: after them the image shows the next row's state
                rows = [frame, min(frame + 1, int(self.ep_len[episode]) - 1)]
        start = int(self.ep_from[episode])
        bits = np.bitwise_or.reduce(np.stack([self.data[start + r, cam] for r in rows]), axis=0)
        bits = np.unpackbits(bits)
        mask = bits[: self.hw[0] * self.hw[1]].reshape(self.hw).astype(np.float32)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
        px = int(round(dilate_px * h / _NATIVE_HW[0]))
        if px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        return mask


# ----------------------------------------------------------------------------------------------
# augmentor


def _to_float(img):
    arr = img.detach().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
    arr = arr.transpose(1, 2, 0).astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return arr


def _to_like(arr, like):
    out = np.clip(arr, 0, 1).transpose(2, 0, 1)
    if torch.is_tensor(like):
        if like.dtype == torch.uint8:
            return torch.from_numpy(np.ascontiguousarray((out * 255).round().astype(np.uint8)))
        return torch.from_numpy(np.ascontiguousarray(out)).to(like.dtype)
    return out


class RandomSceneAugmentor:
    """Callable applied to ``FeatureTransform.pad_and_concat`` output (mutates it in place)."""

    def __init__(self, config):
        self.cfg = load_random_aug_config(config) if isinstance(config, str) else _merge(DEFAULTS, config)
        self._ready = False
        self._seed = None

    # assets are loaded lazily inside each dataloader worker
    def _init(self):
        cfg = self.cfg
        cv2.setNumThreads(1)  # one dataloader worker per core already
        for section, key, used, hint in (
            ("mask", "head_ref", True, "run scripts/random_aug/build_assets.py"),
            ("mask", "arm_masks", True, "run scripts/random_aug/render_arm_masks.py or set mask.arm_masks: null"),
            ("distractor", "library", cfg["distractor"]["prob"] > 0,
             "run scripts/random_aug/build_assets.py or set distractor.prob: 0"),
            ("instruction", "paraphrase_file", cfg["instruction"]["prob"] > 0,
             "run scripts/random_aug/gen_instruction_paraphrases.py or set instruction.prob: 0"),
        ):
            path = cfg[section][key]
            if used and path and not os.path.exists(path):
                raise FileNotFoundError(f"random_aug: {section}.{key}={path} not found ({hint})")
        self.head_ref = None
        if cfg["mask"]["head_ref"]:
            ref = cv2.imread(cfg["mask"]["head_ref"], cv2.IMREAD_COLOR)
            if ref is None:
                raise FileNotFoundError(f"random_aug: head_ref {cfg['mask']['head_ref']} not found")
            self.head_ref = HeadReference(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), cfg["mask"])
        self.arm_masks = ArmMasks(cfg["mask"]["arm_masks"]) if cfg["mask"]["arm_masks"] else None
        self.textures = TextureSource(cfg["background"])
        self.distractors = None
        if cfg["distractor"]["library"] and cfg["distractor"]["prob"] > 0:
            self.distractors = DistractorLibrary(cfg["distractor"]["library"])
        self.paraphrases = {}
        if cfg["instruction"]["paraphrase_file"] and cfg["instruction"]["prob"] > 0:
            with open(cfg["instruction"]["paraphrase_file"]) as f:
                self.paraphrases = {k.strip(): v for k, v in json.load(f).items() if v}
        if self.head_ref is None and (cfg["background"]["prob"] > 0 or cfg["distractor"]["prob"] > 0):
            logger.warning("random_aug: mask.head_ref not set, head camera uses the colour-based mask")
        logger.info(
            f"random_aug: prob={cfg['prob']} arm_masks={self.arm_masks is not None} "
            f"textures={len(self.textures.files)} files + procedural, "
            f"distractors={len(self.distractors) if self.distractors else 0}, paraphrased instructions={len(self.paraphrases)}"
        )
        self._ready = True

    def _get_rng(self):
        # torch seeds every dataloader worker differently (and per epoch); forked workers would
        # otherwise share one numpy state. Those seeds repeat on every rank (same train.seed), so
        # the rank is mixed in; rank 0 keeps the plain seed.
        seed = torch.initial_seed()
        if seed != self._seed:
            self._seed = seed
            rank = int(os.environ.get("RANK", 0))
            self._rng = np.random.default_rng(seed % (2**63) if rank == 0 else [seed % (2**63), rank])
        return self._rng

    def _role(self, key):
        if any(s in key for s in self.cfg["head_keys"]):
            return "head"
        if any(s in key for s in self.cfg["wrist_keys"]):
            return "wrist"
        return None

    @staticmethod
    def _arm_cam(key, role):
        if role == "head":
            return 0
        if "left" in key:
            return 1
        if "right" in key:
            return 2
        return None

    def __call__(self, batch_dict, episode_index=None, frame_index=None, future_offset=None):
        """``frame_index`` / ``future_offset`` (future frame = frame + offset, clamped to the episode)
        select the arm masks; without them the colour rules alone decide."""
        if not self._ready:
            self._init()
        rng = self._get_rng()
        self._augment_instruction(batch_dict, rng)

        images = batch_dict.get("image") or {}
        if not images or rng.random() >= self.cfg["prob"]:
            return batch_dict
        future = batch_dict.get("future_image") or {}
        if torch.is_tensor(episode_index):
            episode_index = int(episode_index.reshape(-1)[0])
        if torch.is_tensor(frame_index):
            frame_index = int(frame_index.reshape(-1)[0])
        frame_ids = [frame_index, None if frame_index is None or future_offset is None else frame_index + future_offset]

        cfg = self.cfg
        scene = {
            "background": rng.random() < cfg["background"]["prob"],
            "distractor": self.distractors is not None and rng.random() < cfg["distractor"]["prob"],
            "geometry": rng.random() < cfg["geometry"]["prob"],
            "light": self._sample_light(rng) if rng.random() < cfg["lighting"]["prob"] else None,
            "same_texture": rng.random() < cfg["background"]["same_texture_prob"],
        }
        for key in list(images.keys()):
            role = self._role(key)
            if role is None:
                continue
            frames = [images[key]] + ([future[key]] if key in future else [])
            arm = (self._arm_cam(key, role), episode_index, frame_ids)
            out = self._augment_camera(frames, role, scene, rng, episode_index, arm)
            images[key] = out[0]
            if len(out) > 1:
                future[key] = out[1]
        return batch_dict

    def _augment_instruction(self, batch_dict, rng):
        icfg = self.cfg["instruction"]
        prompts = batch_dict.get("prompt")
        if not prompts or icfg["prob"] <= 0:
            return
        out = []
        for prompt in prompts:
            if self.paraphrases and rng.random() < icfg["prob"]:
                choices = self.paraphrases.get(prompt.strip())
                if choices:
                    prompt = choices[int(rng.integers(len(choices)))]
            if rng.random() < icfg["format_noise"]:
                if prompt.endswith("."):
                    prompt = prompt[:-1]
                elif prompt[:1].isupper():
                    prompt = prompt[:1].lower() + prompt[1:]
            out.append(prompt)
        batch_dict["prompt"] = out

    def _augment_camera(self, frames, role, scene, rng, episode_index, arm=(None, None, (None, None))):
        arrs = [_to_float(f) for f in frames]
        h, w = arrs[0].shape[:2]
        cfg = self.cfg
        use_bg = scene["background"] and (role == "head" or rng.random() < cfg["background"]["wrist_prob"])
        wrist_p = cfg["distractor"]["wrist_prob"]
        use_distractor = scene["distractor"] and (role == "head" or (wrist_p > 0 and rng.random() < wrist_p))
        if use_bg or use_distractor:
            ref = self.head_ref.at(h, w) if (role == "head" and self.head_ref is not None) else None
            cam, episode, frame_ids = arm
            protect = [
                self.arm_masks.get(episode, f, cam, h, w, cfg["mask"]["arm_dilate_px"])
                if self.arm_masks is not None and cam is not None else None
                for f in frame_ids[: len(arrs)]
            ]
            masks = [compute_background(a, cfg["mask"], ref, p) for a, p in zip(arrs, protect)]
        if use_bg:
            tex_table, tex_wall = self._sample_textures(rng, h, w, role, scene["same_texture"])
            for i, (a, m) in enumerate(zip(arrs, masks)):
                tex = np.where(m["wall"][..., None], tex_wall, tex_table) * m["shade"][..., None]
                alpha = m["alpha"][..., None]
                arrs[i] = a * (1 - alpha) + tex * alpha
        if use_distractor:
            ref = self.head_ref.at(h, w) if (role == "head" and self.head_ref is not None) else None
            plans = self._plan_distractors(rng, masks[0], h, w, episode_index, ref, wrist=role == "wrist")
            for i, m in enumerate(masks):
                arrs[i] = self._paste(arrs[i], plans, m["bg"])
        if role == "head" and scene["geometry"]:
            gcfg = cfg["geometry"]
            s = 1 + rng.uniform(-gcfg["scale"], gcfg["scale"])
            tx, ty = rng.uniform(-gcfg["shift"], gcfg["shift"], 2) * (w, h)
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
            mat[:, 2] += (tx, ty)
            arrs = [cv2.warpAffine(a, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for a in arrs]
        if scene["light"] is not None:
            illum = self._illumination(rng, h, w)
            arrs = [self._apply_light(a, scene["light"], illum) for a in arrs]
        return [_to_like(a, f) for a, f in zip(arrs, frames)]

    def _sample_textures(self, rng, h, w, role, same):
        bcfg = self.cfg["background"]
        if role == "head":
            table = self.textures.sample_table(rng, h, w, bcfg["perspective"])
        else:
            table = self.textures.sample(rng, h, w)
        wall = table if same else self.textures.sample(rng, h, w)
        return table, wall

    def _plan_distractors(self, rng, mask, h, w, episode_index, ref, wrist=False):
        """Pick cut-outs from other tasks and free table positions (in the current frame)."""
        dcfg = self.cfg["distractor"]
        lib = self.distractors
        own_task = None if episode_index is None else episode_index // self.cfg["episodes_per_task"]
        sy, sx = h / lib.native_hw[0], w / lib.native_hw[1]
        horizon = dcfg["horizon"] * h
        clear = max(1, int(round(dcfg["clearance_px"] * sy)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * clear + 1, 2 * clear + 1))
        blocked = cv2.dilate((~mask["bg"]).astype(np.uint8), kernel).astype(bool)
        if wrist:  # no table plane model: standing on the table (below the wall band), enlarged (the camera is close)
            wall_rows = np.flatnonzero(mask["wall"][: h // 2].mean(1) > 0.5)
            table_start = wall_rows.max() + 1 if len(wall_rows) else 0
            num, y_lo, max_frac, top = dcfg["wrist_num"], table_start + 0.03 * h, 0.6, 0
        else:
            table_top = ref["table_top"] if ref is not None else int(0.15 * h)
            num, y_lo, max_frac, top = dcfg["num"], table_top + dcfg["top_margin"] * h, 0.5, table_top
        plans = []
        for _ in range(int(rng.integers(num[0], num[1] + 1))):
            for _ in range(20):
                idx = lib.sample(rng, dcfg["task_balance"])
                if not lib.allowed(idx, own_task):
                    continue
                rgba = lib.get(idx)
                if wrist:  # cy: the bottom (contact) row of the cut-out
                    cy = rng.uniform(y_lo, h)
                    factor = rng.uniform(*dcfg["wrist_scale"])
                else:
                    cy = rng.uniform(y_lo, 0.9 * h)
                    src_y = lib.center_y[idx] * sy
                    factor = np.clip((cy - horizon) / max(src_y - horizon, 1.0), 0.6, 1.6) * rng.uniform(*dcfg["scale"])
                ch = max(4, int(round(rgba.shape[0] * sy * factor)))
                cw = max(4, int(round(rgba.shape[1] * sx * factor)))
                if ch >= h * max_frac or cw >= w * max_frac:
                    continue
                y0 = int(round(cy - ch)) if wrist else int(round(cy - ch / 2))
                x0 = int(rng.integers(0, w - cw))
                if y0 < top or y0 + ch > h:
                    continue
                crop = cv2.resize(rgba, (cw, ch), interpolation=cv2.INTER_AREA)
                region = blocked[y0:y0 + ch, x0:x0 + cw]
                if (region & (crop[..., 3] > 0.1)).any():
                    continue
                plans.append((crop, y0, x0))
                blocked[y0:y0 + ch, x0:x0 + cw] |= crop[..., 3] > 0.1
                break
        return plans

    def _paste(self, img, plans, bg):
        out = img.copy()
        shadow = self.cfg["distractor"]["shadow"]
        for crop, y0, x0 in plans:
            ch, cw = crop.shape[:2]
            free = bg[y0:y0 + ch, x0:x0 + cw].astype(np.float32)[..., None]
            if shadow < 1:
                # soft contact shadow slightly below the object
                sh = cv2.GaussianBlur(crop[..., 3], (0, 0), max(1.0, ch / 12))
                dy = max(1, ch // 12)
                sh = np.roll(sh, dy, axis=0)
                sh[:dy] = 0
                out[y0:y0 + ch, x0:x0 + cw] *= 1 - (1 - shadow) * sh[..., None] * free
            alpha = crop[..., 3:4] * free
            out[y0:y0 + ch, x0:x0 + cw] = out[y0:y0 + ch, x0:x0 + cw] * (1 - alpha) + crop[..., :3] * alpha
        return out

    def _sample_light(self, rng):
        lcfg = self.cfg["lighting"]
        temp = rng.uniform(-1, 1) * lcfg["temperature"]
        gains = np.array([1 + temp, 1 + rng.uniform(-lcfg["tint"], lcfg["tint"]), 1 - temp], np.float32)
        light = {"gain": rng.uniform(*lcfg["gain"]), "gains": gains, "gamma": rng.uniform(*lcfg["gamma"])}
        if rng.random() < lcfg["crazy_prob"]:
            if rng.random() < 0.5:
                light["gains"] = rng.uniform(0.4, 1.6, 3).astype(np.float32)
            else:
                light["gain"] = rng.choice([rng.uniform(0.3, 0.5), rng.uniform(1.6, 2.2)])
        return light

    def _illumination(self, rng, h, w):
        strength = rng.uniform(0, self.cfg["lighting"]["spatial"])
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        if rng.random() < 0.5:  # linear falloff
            theta = rng.random() * 2 * np.pi
            t = ((xx / w - 0.5) * np.cos(theta) + (yy / h - 0.5) * np.sin(theta)) * 2
        else:  # spot light
            cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.3, 1.0) * max(h, w)
            t = 2 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) - 1
        return (1 + strength / 2 * t)[..., None]

    @staticmethod
    def _apply_light(img, light, illum):
        img = np.clip(img, 0, 1) ** light["gamma"]
        return img * light["gains"] * light["gain"] * illum
