"""Frame access to the clean LeRobot v3 training set (shared by the random_aug scripts)."""

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lingbot-vla-v2"))

DEFAULT_DATA = REPO / "data/training_data/RoboTwin_lerobot_v30"  # fixed by scripts/fix_v30_videos.py
HEAD_KEY = "observation.images.cam_high"
CAM_KEYS = (HEAD_KEY, "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")


class CleanFrames:
    def __init__(self, root=DEFAULT_DATA):
        self.root = Path(root).expanduser()
        self.fps = json.loads((self.root / "meta/info.json").read_text())["fps"]
        files = sorted(glob.glob(str(self.root / "meta/episodes/*/*.parquet")))
        self.episodes = pd.concat([pd.read_parquet(f) for f in files]).set_index("episode_index").sort_index()

    def __len__(self):
        return len(self.episodes)

    def length(self, ep):
        return int(self.episodes.loc[ep, "length"])

    def video_length(self, ep, key=HEAD_KEY):
        """Frames the episode's video segment really has: ``length`` on the fixed data
        (scripts/fix_v30_videos.py); the official v3.0 release misses 1-2 frames mid-episode in ~25%
        of the episodes, and the rows past it decode as the next episode's first frames."""
        row = self.episodes.loc[ep]
        span = float(row[f"videos/{key}/to_timestamp"]) - float(row[f"videos/{key}/from_timestamp"])
        return min(self.length(ep), round(span * self.fps))

    def instruction(self, ep):
        return str(self.episodes.loc[ep, "tasks"][0])

    def read(self, requests):
        """requests: [(episode, frame_index, camera_key)] -> [uint8 HxWx3 RGB at native resolution]."""
        by_file = defaultdict(list)
        for i, (ep, frame, key) in enumerate(requests):
            row = self.episodes.loc[ep]
            path = self.root / "videos" / key / (
                f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}/file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
            )
            by_file[path].append((i, round(float(row[f"videos/{key}/from_timestamp"]) * self.fps) + frame))
        out = [None] * len(requests)
        for path, items in by_file.items():
            decoder = VideoDecoder(str(path), dimension_order="NHWC")
            last = decoder.metadata.num_frames - 1
            items.sort(key=lambda x: x[1])
            # by frame index: the pts of a file's last frame can sit exactly at the stream end
            frames = decoder.get_frames_at(indices=[min(f, last) for _, f in items]).data.numpy()
            for (i, _), frame in zip(items, frames):
                out[i] = np.ascontiguousarray(frame)
        return out
