"""Rebuild the videos of the official RoboTwin_lerobot_v30 clean dataset from the official v2.1 archive.

The official v2.1 -> v3.0 conversion lost 1-2 frames mid-episode in ~25% of the episodes (e.g.
episode 0 cam_high misses frame 31): the mp4 holds fewer frames than the parquet rows, so later
images lead the state / action by one frame and the last rows decode as the next episode's first
frames. The v2.1 archive (one mp4 per episode) is intact and encoded the same way (AV1, GOP 2), so
its packets are copied losslessly into the v3.0 file layout. Parquet data are kept; only the video
files and the episodes' video from/to timestamps change. The two archives order the episodes
differently, so episodes are matched by their state / action arrays.

    hf download TianxingChen/RoboTwin2.0 lerobot_dataset/RoboTwin_lerobot_v21.zip --repo-type dataset --local-dir data
    mv data/training_data/RoboTwin_lerobot_v30 data/training_data/RoboTwin_lerobot_v30_old
    python scripts/fix_v30_videos.py          # _old -> data/training_data/RoboTwin_lerobot_v30
"""

import argparse
import glob
import io
import json
import shutil
import zipfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]


def arrays(df):
    df = df.sort_values("frame_index")
    return np.concatenate([np.stack(df["observation.state"].values), np.stack(df["action"].values)], 1).astype(np.float32)


def match_episodes(v30, zf, root21):
    """v3.0 episode index -> v2.1 parquet / video name stem, matched by identical state + action."""
    data30 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(v30 / "data/*/*.parquet")))])
    eps30 = {int(ep): arrays(df) for ep, df in data30.groupby("episode_index")}
    names21 = sorted(n for n in zf.namelist() if n.startswith(f"{root21}data/") and n.endswith(".parquet"))
    by_len = {}
    for name in names21:
        a = arrays(pd.read_parquet(io.BytesIO(zf.read(name))))
        by_len.setdefault(len(a), []).append((name, a))
    mapping, used = {}, set()
    for ep, a in eps30.items():
        best, err = None, np.inf
        for name, b in by_len.get(len(a), []):
            e = float(np.abs(a - b).max())
            if e < err:
                best, err = name, e
        if best is None or err > 1e-6 or best in used:
            raise RuntimeError(f"v3.0 episode {ep}: no unique v2.1 match (best {best}, max diff {err})")
        used.add(best)
        mapping[ep] = best
    return mapping


def write_video(out_path, members, lengths, zf, fps):
    """Concatenate the packets of the v2.1 episode videos ``members`` into one mp4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out_path), "w", format="mp4") as out:
        ostream, offset = None, 0
        for name, n in zip(members, lengths):
            with av.open(io.BytesIO(zf.read(name))) as src:
                s = src.streams.video[0]
                if ostream is None:
                    ostream = out.add_stream_from_template(s, opaque=True)  # remux, no encoder
                    tb = ostream.time_base = s.time_base
                step = Fraction(1, fps) / tb
                assert s.time_base == tb and step.denominator == 1, (name, s.time_base)
                count = 0
                for pkt in src.demux(s):
                    if pkt.pts is None:  # flush packet
                        continue
                    assert pkt.pts == pkt.dts == count * step, (name, count, pkt.pts, pkt.dts)
                    pkt.pts = pkt.dts = (offset + count) * int(step)
                    pkt.time_base = tb
                    pkt.stream = ostream
                    out.mux(pkt)
                    count += 1
                if count != n:
                    raise RuntimeError(f"{name}: {count} frames, parquet has {n} rows")
            offset += n


def verify(out_dir, zf, mapping, cams, video21, short, num, seed):
    """Decode by timestamp (as training does) and compare with the v2.1 episode videos; half of the
    checked episodes are ones whose original v3.0 video was short."""
    from torchcodec.decoders import VideoDecoder

    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(out_dir / "meta/episodes/*/*.parquet")))])
    eps = eps.set_index("episode_index")
    fps = json.loads((out_dir / "meta/info.json").read_text())["fps"]
    rng = np.random.default_rng(seed)
    checked = 0
    for key in cams:
        bad = rng.permutation(short[key])[: num // 2].tolist()
        picks = bad + rng.choice(eps.index.values, num - len(bad), replace=False).tolist()
        for ep in picks:
            ep = int(ep)
            row = eps.loc[ep]
            n = int(row["length"])
            path = out_dir / "videos" / key / f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}/file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
            frames = sorted({0, int(rng.integers(0, n)), n - 1})
            # same lookup as training (video_utils.decode_video_frames_torchcodec): round(ts * fps)
            idx = [round((float(row[f"videos/{key}/from_timestamp"]) + t / fps) * fps) for t in frames]
            new = VideoDecoder(str(path), dimension_order="NHWC").get_frames_at(idx).data.numpy()
            ref = VideoDecoder(zf.read(video21(mapping[ep], key)), dimension_order="NHWC").get_frames_at(frames).data.numpy()
            if not np.array_equal(new, ref):
                raise RuntimeError(f"episode {ep} {key} frames {frames}: decoded frames differ from v2.1")
            checked += len(frames)
    print(f"verify: {checked} frames identical to v2.1 (first / random / last frame of {num} episodes per camera, "
          f"{sum(min(len(v), num // 2) for v in short.values())} of them originally short)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v30", default=str(REPO / "data/training_data/RoboTwin_lerobot_v30_old"))
    p.add_argument("--v21_zip", default=str(REPO / "data/lerobot_dataset/RoboTwin_lerobot_v21.zip"))
    p.add_argument("--out", default=str(REPO / "data/training_data/RoboTwin_lerobot_v30"))
    p.add_argument("--verify", type=int, default=100, help="episodes per camera to compare against v2.1")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    v30, out_dir = Path(a.v30), Path(a.out)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; remove it first")

    info = json.loads((v30 / "meta/info.json").read_text())
    fps = info["fps"]
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    zf = zipfile.ZipFile(a.v21_zip)
    root21 = zf.namelist()[0].split("/")[0] + "/"
    video21 = lambda parquet_name, key: parquet_name.replace(f"{root21}data/", f"{root21}videos/").replace(
        "/episode_", f"/{key}/episode_").replace(".parquet", ".mp4")

    mapping = match_episodes(v30, zf, root21)
    print(f"matched {len(mapping)} episodes (v3.0 -> v2.1)")

    ep_files = sorted(glob.glob(str(v30 / "meta/episodes/*/*.parquet")))
    eps = pd.concat([pd.read_parquet(f) for f in ep_files])
    new_ts = {}
    for key in cams:
        groups = eps.groupby([f"videos/{key}/chunk_index", f"videos/{key}/file_index"])
        for (chunk, file), g in groups:
            g = g.sort_values(f"videos/{key}/from_timestamp")
            path = out_dir / "videos" / key / f"chunk-{int(chunk):03d}/file-{int(file):03d}.mp4"
            lengths = g["length"].astype(int).tolist()
            write_video(path, [video21(mapping[int(e)], key) for e in g["episode_index"]], lengths, zf, fps)
            start = np.concatenate([[0], np.cumsum(lengths)[:-1]])
            for e, s0, n in zip(g["episode_index"], start, lengths):
                new_ts[(int(e), key)] = (s0 / fps, (s0 + n) / fps)
        print(f"{key}: {len(groups)} files written")

    shutil.copytree(v30 / "data", out_dir / "data")
    (out_dir / "meta").mkdir(exist_ok=True)
    for f in ("info.json", "stats.json", "tasks.parquet"):
        shutil.copy2(v30 / "meta" / f, out_dir / "meta" / f)
    for f in ep_files:
        table = pq.read_table(f)  # keep the original schema, replace only the video timestamps
        episodes = table.column("episode_index").to_pylist()
        for key in cams:
            for i, col in enumerate(("from_timestamp", "to_timestamp")):
                name = f"videos/{key}/{col}"
                values = pa.array([new_ts[(e, key)][i] for e in episodes], type=table.schema.field(name).type)
                table = table.set_column(table.schema.get_field_index(name), name, values)
        dst = out_dir / Path(f).relative_to(v30)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst)
    print(f"saved {out_dir}")

    if a.verify:
        span = lambda key: (eps[f"videos/{key}/to_timestamp"] - eps[f"videos/{key}/from_timestamp"]) * fps
        short = {key: eps["episode_index"][np.round(span(key)).astype(int) < eps["length"]].astype(int).tolist() for key in cams}
        verify(out_dir, zf, mapping, cams, video21, short, a.verify, a.seed)


if __name__ == "__main__":
    main()
