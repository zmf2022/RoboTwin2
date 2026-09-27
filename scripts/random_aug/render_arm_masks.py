"""Render robot-arm masks for every clean training frame (head + both wrist cameras).

The white arm links look like the white table to any colour rule, so random_aug would texture them.
The clean data records the joint states and the cameras / robot are fixed by the RoboTwin config,
so the simulator renders the exact arm silhouette per frame (robot + table + wall only, no objects,
no randomization, segmentation only). Stored half resolution, bit-packed:
``arm_masks.npy`` uint8 [num_frames, 3 (head, left wrist, right wrist), 120*160/8] indexed by the
dataset's global frame index, plus ``arm_masks_index.npz`` (episode start index / length, and per
camera the frames the episode's video segment really has: the full length on the fixed data; in
the official unfixed v3.0, ~25% of the episodes miss 1-2 frames mid-episode, so later images lead
the state by a frame and the last rows decode as the next episode's first frames).

    python scripts/random_aug/render_arm_masks.py --workers 4
    python scripts/random_aug/render_arm_masks.py --verify 8 --verify_out arm_overlay.png   # alignment check
    python scripts/random_aug/render_arm_masks.py --index_only   # rewrite arm_masks_index.npz only
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import CAM_KEYS, DEFAULT_DATA, REPO, CleanFrames

MASK_HW = (120, 160)
CAMS = ("head_camera", "left_camera", "right_camera")


def load_states(root):
    files = sorted(glob.glob(str(Path(root).expanduser() / "data/*/*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["index", "episode_index", "frame_index", "observation.state"]) for f in files])
    df = df.sort_values("index")
    assert (df["index"].values == np.arange(len(df))).all(), "global frame index is not contiguous"
    return np.stack(df["observation.state"].values).astype(np.float32), df["episode_index"].values


class ArmRenderer:
    def __init__(self, task_config="demo_clean"):
        os.chdir(REPO)  # embodiment paths in the RoboTwin config are relative to the repo root
        sys.path.insert(0, str(REPO))
        import sapien
        import yaml
        from envs import CONFIGS_PATH
        from envs.camera import Camera
        from envs.robot import Robot
        from envs.utils import create_box, create_table

        with open(Path(CONFIGS_PATH) / f"{task_config}.yml") as f:
            args = yaml.safe_load(f)
        with open(Path(CONFIGS_PATH) / "_embodiment_config.yml") as f:
            robot_file = yaml.safe_load(f)[args["embodiment"][0]]["file_path"]
        with open(Path(robot_file) / "config.yml") as f:
            robot_cfg = yaml.safe_load(f)
        args.update(left_robot_file=robot_file, right_robot_file=robot_file, dual_arm_embodied=True,
                    left_embodiment_config=robot_cfg, right_embodiment_config=robot_cfg)

        engine = sapien.Engine()
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)
        sapien.render.set_camera_shader_dir("default")  # rasterization is enough for segmentation
        self.scene = engine.create_scene(sapien.SceneConfig())
        self.scene.add_ground(0)
        self.scene.default_physical_material = self.scene.create_physical_material(0.5, 0.5, 0)
        # same table / wall as Base_Task.create_table_and_wall (clean: no height bias); they occlude
        # the arm parts below / behind them
        create_box(self.scene, sapien.Pose(p=[0, 1, 1.5]), half_size=[3, 0.6, 1.5], color=(1, 0.9, 0.9),
                   name="wall", is_static=True)
        create_table(self.scene, sapien.Pose(p=[0, 0, 0.74]), length=1.2, width=0.7, height=0.74,
                     thickness=0.05, is_static=True)
        self.robot = Robot(self.scene, need_topp=False, **args)
        self.robot.init_joints()
        self.cameras = Camera(bias=0, random_head_camera_dis=0, **args)
        self.cameras.load_camera(self.scene)
        head = self.cameras.static_camera_list[self.cameras.static_camera_name.index("head_camera")]
        self.cams = [head, self.cameras.left_camera, self.cameras.right_camera]

        r = self.robot
        entities = {id(r.left_entity): r.left_entity, id(r.right_entity): r.right_entity}.values()
        self.robot_ids = np.array(sorted({link.entity.per_scene_id for e in entities for link in e.get_links()}))
        self.arms = []
        for entity, joints, gripper, scale in (
            (r.left_entity, r.left_arm_joints, r.left_gripper, r.left_gripper_scale),
            (r.right_entity, r.right_arm_joints, r.right_gripper, r.right_gripper_scale),
        ):
            active = entity.get_active_joints()
            self.arms.append((entity, [active.index(j) for j in joints],
                              [(active.index(g[0]), g[1], g[2]) for g in gripper], scale))

    def render(self, state):
        """state: 14 = left 6 joints + gripper [0, 1], right 6 joints + gripper -> bool [3, 240, 320]."""
        qpos = {}
        for arm, (entity, joint_idx, gripper, scale) in enumerate(self.arms):
            q = qpos.setdefault(id(entity), entity.get_qpos().copy())
            s = state[7 * arm:7 * arm + 7]
            q[joint_idx] = s[:6]
            real = scale[0] + np.clip(s[6], 0, 1) * (scale[1] - scale[0])
            for idx, mult, offset in gripper:
                q[idx] = real * mult + offset
        for entity, _, _, _ in self.arms:
            entity.set_qpos(qpos[id(entity)])
        self.cameras.update_wrist_camera(self.robot.left_camera.get_pose(), self.robot.right_camera.get_pose())
        self.scene.update_render()
        out = []
        for cam in self.cams:
            cam.take_picture()
            seg = cam.get_picture("Segmentation")[..., 1]
            out.append(np.isin(seg, self.robot_ids))
        return np.stack(out)


def pack(masks):
    import cv2

    small = [cv2.resize(m.astype(np.uint8), MASK_HW[::-1], interpolation=cv2.INTER_AREA) > 0 for m in masks]
    return np.packbits(np.stack(small).reshape(len(masks), -1), axis=1)


def write_index(data, out):
    frames = CleanFrames(data)
    e, fps = frames.episodes, frames.fps
    video_len, next_same_file = [], []
    for key in CAM_KEYS:
        span = e[f"videos/{key}/to_timestamp"].values - e[f"videos/{key}/from_timestamp"].values
        video_len.append(np.round(span * fps).astype(np.int64))
        file_id = e[f"videos/{key}/chunk_index"].values * 100000 + e[f"videos/{key}/file_index"].values
        next_same_file.append(np.append(file_id[1:] == file_id[:-1], False))
    np.savez(Path(out).with_name(Path(out).stem + "_index.npz"),
             ep_from=e["dataset_from_index"].values.astype(np.int64),
             ep_len=e["length"].values.astype(np.int64),
             video_len=np.stack(video_len, 1), next_same_file=np.stack(next_same_file, 1),
             mask_hw=np.array(MASK_HW), cams=np.array(CAMS))


def worker(rank, num, states, out_path):
    renderer = ArmRenderer()
    out = np.load(out_path, mmap_mode="r+")
    rows = np.arange(rank, len(states), num)
    for k, i in enumerate(rows):
        out[i] = pack(renderer.render(states[i]))
        if rank == 0 and k % 2000 == 0:
            print(f"rank0 {k}/{len(rows)}", flush=True)
    out.flush()


def verify(a, states):
    import cv2

    frames = CleanFrames(a.data)
    renderer = ArmRenderer()
    rng = np.random.default_rng(0)
    starts = frames.episodes["dataset_from_index"].values
    rows = []
    for ep in rng.choice(len(frames), a.verify, replace=False):
        t = int(rng.integers(0, frames.length(int(ep))))
        imgs = frames.read([(int(ep), t, k) for k in CAM_KEYS])
        masks = renderer.render(states[starts[ep] + t])
        tiles = []
        for img, m in zip(imgs, masks):
            vis = img.astype(np.float32)
            vis[m] = vis[m] * 0.5 + np.array([255, 0, 255]) * 0.5
            tiles.append(vis.astype(np.uint8))
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(a.verify_out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f"saved {a.verify_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--out", default=str(REPO / "data/random_aug/arm_masks.npy"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--verify", type=int, default=0, help="only render N random frames as an overlay check")
    p.add_argument("--verify_out", default="arm_overlay.png")
    p.add_argument("--index_only", action="store_true")
    a = p.parse_args()
    a.data, a.out, a.verify_out = (str(Path(x).expanduser().resolve()) for x in (a.data, a.out, a.verify_out))

    if a.index_only:
        write_index(a.data, a.out)
        return
    states, episode = load_states(a.data)
    if a.verify:
        verify(a, states)
        return
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(states), 3, MASK_HW[0] * MASK_HW[1] // 8))
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, a.workers, states, str(out))) for r in range(a.workers)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
        assert proc.exitcode == 0, f"render worker failed ({proc.exitcode})"
    write_index(a.data, out)
    print(f"saved {out} ({len(states)} frames)")


if __name__ == "__main__":
    main()
