"""RoboTwin sim client for a LingBot-VLA 2.0 websocket policy server.

Port of lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py to this repo's
layout (env_cfg/task_config). Evaluation protocol is unchanged: seeds from 100000*(1+seed),
expert-check each seed, 100 successful-seed episodes per task.
"""
import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "description" / "utils", ROOT / "lingbot-vla-v2"):
    sys.path.insert(0, str(p))
os.chdir(ROOT)

from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
from deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


def load_task_args(task_name, task_config):
    with open(Path(CONFIGS_PATH) / f"{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config

    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    with open(Path(CONFIGS_PATH) / "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)

    head = camera_config[args["camera"]["head_camera_type"]]
    args["head_camera_h"], args["head_camera_w"] = head["h"], head["w"]

    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError("only single-embodiment configs are supported")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    args["left_robot_file"] = args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as f:
        robot_cfg = yaml.safe_load(f)
    args["left_embodiment_config"] = args["right_embodiment_config"] = robot_cfg
    return args, head


def run_episode(env, model, robo_name):
    model.infer(dict(reset=True, robo_name=robo_name, path_to_pi_model=None))
    while env.take_action_cnt < env.step_lim:
        obs = env.get_obs()
        ret = model.infer({
            "observation.images.cam_high": obs["observation"]["head_camera"]["rgb"],
            "observation.images.cam_left_wrist": obs["observation"]["left_camera"]["rgb"],
            "observation.images.cam_right_wrist": obs["observation"]["right_camera"]["rgb"],
            "observation.state": obs["joint_action"]["vector"],
            "task": env.get_instruction(),
        })
        action = ret["action"]
        for act in (action if action.ndim == 2 else [action]):
            env.take_action(act)
            if env.eval_success:
                return True
    return False


def main(a):
    args, head = load_task_args(a.task_name, a.task_config)
    args["eval_mode"] = True
    args["eval_video_log"] = a.video
    out_dir = Path(a.output_dir) / a.task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.video:
        args["eval_video_save_dir"] = out_dir
    video_size = f"{head['w']}x{head['h']}"

    env = getattr(__import__(f"envs.{a.task_name}", fromlist=[a.task_name]), a.task_name)()
    model = WebsocketClientPolicy(port=a.port)

    seed = 100000 * (1 + a.seed)
    suc = done = ep_id = 0
    clear_cache_freq = args["clear_cache_freq"]
    render_freq = args["render_freq"]
    while done < a.test_num:
        # expert check: skip seeds the scripted planner cannot solve
        args["render_freq"] = 0
        try:
            env.setup_demo(now_ep_num=ep_id, seed=seed, is_test=True, **args)
            info = env.play_once()
            env.close_env()
        except UnStableError:
            env.close_env()
            seed += 1
            continue
        except Exception as e:
            env.close_env()
            print(f"error occurs ! {e}")
            seed += 1
            continue
        args["render_freq"] = render_freq
        if not (env.plan_success and env.check_success()):
            seed += 1
            continue

        try:
            env.setup_demo(now_ep_num=ep_id, seed=seed, is_test=True, **args)
        except Exception as e:
            env.close_env()
            print(f"setup failed on seed {seed}: {e!r}")
            seed += 1
            continue
        # setup_demo seeds np.random; the instruction generator also uses python `random`, seed it too
        random.seed(seed)
        desc = generate_episode_descriptions(a.task_name, [info["info"]], a.test_num)
        env.set_instruction(instruction=np.random.choice(desc[0][a.instruction_type]))

        ffmpeg = None
        if env.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                 "-video_size", video_size, "-framerate", str(a.video_fps), "-i", "-",
                 "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                 f"{env.eval_video_path}/episode{done}.mp4"],
                stdin=subprocess.PIPE)
            env._set_eval_video_ffmpeg(ffmpeg)

        succ = run_episode(env, model, a.robo_name)

        if ffmpeg is not None:
            env._del_eval_video_ffmpeg()
            tag = "success" if succ else "failure"
            src = Path(env.eval_video_path) / f"episode{done}.mp4"
            if src.exists():
                src.rename(src.with_name(f"episode{done}_{tag}.mp4"))

        suc += int(succ)
        done += 1
        ep_id += 1
        env.close_env(clear_cache=(done % clear_cache_freq == 0))
        print(f"{a.task_name} | {a.task_config}\nSuccess rate: {suc}/{done} => {round(suc / done * 100, 1)}%, current seed: {seed}\n",
              flush=True)
        seed += 1

    (out_dir / "result.json").write_text(json.dumps({
        "task": a.task_name, "task_config": a.task_config, "instruction_type": a.instruction_type,
        "attempts": done, "successes": suc, "time": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", required=True)
    p.add_argument("--task_config", default="demo_clean")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_num", type=int, default=100)
    # LingBot official client uses "unseen" for both settings (policy/ACT/deploy_policy.yml @13c3c47)
    p.add_argument("--instruction_type", default="unseen", choices=["seen", "unseen"])
    p.add_argument("--robo_name", default="robotwin")
    p.add_argument("--video", type=int, default=1)
    p.add_argument("--video_fps", type=int, default=10)
    main(p.parse_args())
