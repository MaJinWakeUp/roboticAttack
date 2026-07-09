"""
Run OpenVLA in a real-world Bridge V2 / WidowX environment with a small Tk GUI.

This is not a simulator viewer. It connects to the WidowX service, shows the live
camera observation, and lets the operator reset, start, pause, step, and stop an
OpenVLA-controlled episode.
"""

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BOUNDS = [
    [0.1, -0.20, -0.01, -1.57, 0],
    [0.45, 0.25, 0.30, 1.57, 0],
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", default="openvla/openvla-7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host_ip", default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--camera_topic", default="/blue/image_raw")
    parser.add_argument("--task", default="")
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--control_frequency", type=float, default=5.0)
    parser.add_argument("--init_ee_pos", type=float, nargs=3, default=[0.3, -0.09, 0.26])
    parser.add_argument("--init_ee_quat", type=float, nargs=4, default=[0, -0.259, 0, -0.966])
    parser.add_argument("--blocking", action="store_true")
    parser.add_argument("--save_data", action="store_true")
    parser.add_argument("--no-save_video", dest="save_video", action="store_false")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--image_scale", type=float, default=0.75)
    parser.add_argument("--manual_translation_step", type=float, default=0.01)
    parser.add_argument("--manual_rotation_step", type=float, default=0.05)
    parser.add_argument("--manual_gripper", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.set_defaults(save_video=True)
    return parser.parse_args()


def make_cfg(args):
    cfg = SimpleNamespace()
    cfg.model_family = "openvla"
    cfg.pretrained_checkpoint = args.pretrained_checkpoint
    cfg.load_in_8bit = args.load_in_8bit
    cfg.load_in_4bit = args.load_in_4bit
    cfg.center_crop = False
    cfg.unnorm_key = "bridge_orig"

    cfg.host_ip = args.host_ip
    cfg.port = args.port
    cfg.init_ee_pos = args.init_ee_pos
    cfg.init_ee_quat = args.init_ee_quat
    cfg.bounds = DEFAULT_BOUNDS
    cfg.camera_topics = [{"name": args.camera_topic}]
    cfg.blocking = args.blocking
    cfg.max_episodes = args.max_episodes
    cfg.max_steps = args.max_steps
    cfg.control_frequency = args.control_frequency
    cfg.save_data = args.save_data
    return cfg


def check_unnorm_key(cfg, model):
    if not hasattr(model, "norm_stats"):
        return
    if cfg.unnorm_key not in model.norm_stats:
        raise ValueError(f"Action un-norm key {cfg.unnorm_key} not found in VLA norm_stats.")


class BridgeGui:
    def __init__(self, root, args, cfg, model, processor, env):
        self.root = root
        self.args = args
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.env = env
        self.resize_size = get_image_resize_size(cfg)

        self.obs = None
        self.running = False
        self.after_id = None
        self.episode_idx = 0
        self.step_idx = 0
        self.last_action = None
        self.photo = None

        self.replay_images = []
        self.rollout_images = []
        self.rollout_states = []
        self.rollout_actions = []

        self.task_var = tk.StringVar(value=args.task)
        self.status_var = tk.StringVar(value="Ready")
        self.action_var = tk.StringVar(value="No action yet")
        self.manual_translation_var = tk.DoubleVar(value=args.manual_translation_step)
        self.manual_rotation_var = tk.DoubleVar(value=args.manual_rotation_step)
        self.manual_gripper_var = tk.DoubleVar(value=args.manual_gripper)

        self.build_ui()

    def build_ui(self):
        self.root.title("Bridge V2 OpenVLA GUI")
        self.root.geometry("820x820")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(frame, text=f"Bridge V2 / {self.cfg.host_ip}:{self.cfg.port}").grid(
            row=0, column=0, columnspan=6, sticky="w"
        )
        ttk.Label(frame, text="Task").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.task_var).grid(row=1, column=1, columnspan=5, sticky="ew", pady=(8, 0))

        self.image_label = ttk.Label(frame)
        self.image_label.grid(row=2, column=0, columnspan=6, pady=10)

        ttk.Button(frame, text="Reset", command=self.reset_episode).grid(row=3, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(frame, text="Start", command=self.start_episode).grid(row=3, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(frame, text="Pause", command=self.pause_episode).grid(row=3, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(frame, text="Step Once", command=self.step_once).grid(row=3, column=3, padx=4, pady=4, sticky="ew")
        ttk.Button(frame, text="Stop", command=lambda: self.finish_episode("stopped")).grid(
            row=3, column=4, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Quit", command=self.close).grid(row=3, column=5, padx=4, pady=4, sticky="ew")

        ttk.Button(frame, text="Mark Success", command=lambda: self.finish_episode("manual success")).grid(
            row=4, column=0, columnspan=3, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Mark Failure", command=lambda: self.finish_episode("manual failure")).grid(
            row=4, column=3, columnspan=3, padx=4, pady=4, sticky="ew"
        )

        ttk.Separator(frame).grid(row=5, column=0, columnspan=6, pady=8, sticky="ew")
        ttk.Label(frame, textvariable=self.status_var, wraplength=780, justify="left").grid(
            row=6, column=0, columnspan=6, sticky="w"
        )
        ttk.Label(frame, textvariable=self.action_var, wraplength=780, justify="left").grid(
            row=7, column=0, columnspan=6, sticky="w", pady=(6, 0)
        )

        ttk.Separator(frame).grid(row=8, column=0, columnspan=6, pady=8, sticky="ew")
        ttk.Label(frame, text="Manual nudge").grid(row=9, column=0, columnspan=6, sticky="w")

        ttk.Button(frame, text="X+", command=lambda: self.manual_axis_step(0, 1)).grid(
            row=10, column=0, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="X-", command=lambda: self.manual_axis_step(0, -1)).grid(
            row=10, column=1, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Y+", command=lambda: self.manual_axis_step(1, 1)).grid(
            row=10, column=2, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Y-", command=lambda: self.manual_axis_step(1, -1)).grid(
            row=10, column=3, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Z+", command=lambda: self.manual_axis_step(2, 1)).grid(
            row=10, column=4, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Z-", command=lambda: self.manual_axis_step(2, -1)).grid(
            row=10, column=5, padx=4, pady=4, sticky="ew"
        )

        ttk.Button(frame, text="Roll+", command=lambda: self.manual_axis_step(3, 1, rotate=True)).grid(
            row=11, column=0, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Roll-", command=lambda: self.manual_axis_step(3, -1, rotate=True)).grid(
            row=11, column=1, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Pitch+", command=lambda: self.manual_axis_step(4, 1, rotate=True)).grid(
            row=11, column=2, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Pitch-", command=lambda: self.manual_axis_step(4, -1, rotate=True)).grid(
            row=11, column=3, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Yaw+", command=lambda: self.manual_axis_step(5, 1, rotate=True)).grid(
            row=11, column=4, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Yaw-", command=lambda: self.manual_axis_step(5, -1, rotate=True)).grid(
            row=11, column=5, padx=4, pady=4, sticky="ew"
        )

        ttk.Button(frame, text="Open", command=lambda: self.manual_gripper_step(1.0)).grid(
            row=12, column=0, columnspan=2, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="Close", command=lambda: self.manual_gripper_step(0.0)).grid(
            row=12, column=2, columnspan=2, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(frame, text="No-op", command=lambda: self.manual_action(np.array([0, 0, 0, 0, 0, 0, self.manual_gripper_var.get()], dtype=np.float32))).grid(
            row=12, column=4, columnspan=2, padx=4, pady=4, sticky="ew"
        )

        ttk.Label(frame, text="Move step").grid(row=13, column=0, sticky="w")
        ttk.Scale(frame, variable=self.manual_translation_var, from_=0.001, to=0.05, orient="horizontal").grid(
            row=13, column=1, columnspan=2, sticky="ew"
        )
        ttk.Label(frame, text="Rotate step").grid(row=13, column=3, sticky="w")
        ttk.Scale(frame, variable=self.manual_rotation_var, from_=0.005, to=0.25, orient="horizontal").grid(
            row=13, column=4, columnspan=2, sticky="ew"
        )
        ttk.Label(frame, text="Gripper cmd").grid(row=14, column=0, sticky="w")
        ttk.Scale(frame, variable=self.manual_gripper_var, from_=0.0, to=1.0, orient="horizontal").grid(
            row=14, column=1, columnspan=5, sticky="ew"
        )

        for column in range(6):
            frame.columnconfigure(column, weight=1)

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def update_image(self, image_array):
        if image_array is None:
            return
        image = Image.fromarray(np.asarray(image_array)).convert("RGB")
        if self.args.image_scale != 1.0:
            width = max(1, int(image.width * self.args.image_scale))
            height = max(1, int(image.height * self.args.image_scale))
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)
        self.root.update_idletasks()

    def reset_episode(self):
        self.pause_episode()
        self.step_idx = 0
        self.replay_images = []
        self.rollout_images = []
        self.rollout_states = []
        self.rollout_actions = []

        self.set_status("Resetting robot. Watch the terminal for start-position prompts.")
        self.obs, _ = self.env.reset()
        self.update_image(self.obs.get("full_image"))
        self.set_status(f"Episode {self.episode_idx + 1} reset. Ready.")

    def start_episode(self):
        if self.episode_idx >= self.cfg.max_episodes:
            self.set_status(f"Reached max_episodes={self.cfg.max_episodes}.")
            return
        task_label = self.task_var.get().strip()
        if not task_label:
            self.set_status("Enter a task prompt before starting.")
            return
        if self.obs is None:
            self.reset_episode()
        self.running = True
        self.set_status(f"Running episode {self.episode_idx + 1}: {task_label}")
        self.schedule_next_step(delay_ms=0)

    def pause_episode(self):
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.set_status("Paused")

    def step_once(self):
        if self.running:
            self.set_status("Pause before using Step Once.")
            return
        if self.episode_idx >= self.cfg.max_episodes:
            self.set_status(f"Reached max_episodes={self.cfg.max_episodes}.")
            return
        if self.obs is None:
            self.reset_episode()
        self.run_policy_step()

    def schedule_next_step(self, delay_ms=None):
        if not self.running:
            return
        if delay_ms is None:
            delay_ms = max(1, int(1000.0 / self.cfg.control_frequency))
        self.after_id = self.root.after(delay_ms, self.policy_tick)

    def policy_tick(self):
        self.after_id = None
        if not self.running:
            return
        self.run_policy_step()
        if self.running:
            self.schedule_next_step()

    def run_policy_step(self):
        task_label = self.task_var.get().strip()
        if not task_label:
            self.running = False
            self.set_status("Enter a task prompt before running.")
            return
        if self.step_idx >= self.cfg.max_steps:
            self.finish_episode("max steps reached")
            return

        started = time.time()
        obs = refresh_obs(self.obs, self.env)
        self.obs = obs
        raw_image = np.asarray(obs["full_image"]).copy()
        self.update_image(raw_image)
        self.replay_images.append(raw_image)

        model_obs = {
            "full_image": get_preprocessed_image({"full_image": raw_image.copy()}, self.resize_size),
            "proprio": obs["proprio"],
        }

        action = get_action(
            self.cfg,
            self.model,
            model_obs,
            task_label,
            processor=self.processor,
            DEVICE=self.args.device,
        )

        if self.cfg.save_data:
            self.rollout_images.append(model_obs["full_image"])
            self.rollout_states.append(obs["proprio"])
            self.rollout_actions.append(action)

        self.last_action = action
        self.action_var.set(f"action={np.array2string(np.round(action, 3), precision=3, separator=', ')}")
        print(f"step={self.step_idx + 1} action={action}", flush=True)

        next_obs, _, _, truncated, _ = self.env.step(action)
        self.obs = next_obs
        if next_obs is not None and "full_image" in next_obs:
            manual_image = np.asarray(next_obs["full_image"]).copy()
            self.update_image(manual_image)
            self.replay_images.append(manual_image)
            if self.cfg.save_data:
                self.rollout_images.append(
                    get_preprocessed_image({"full_image": manual_image.copy()}, self.resize_size)
                )
                self.rollout_states.append(next_obs["proprio"])
                self.rollout_actions.append(action)

        self.step_idx += 1
        elapsed = time.time() - started
        self.set_status(
            f"episode={self.episode_idx + 1} step={self.step_idx}/{self.cfg.max_steps} "
            f"elapsed={elapsed:.2f}s truncated={truncated}"
        )

        if truncated:
            self.finish_episode("environment truncated")

    def manual_axis_step(self, axis, direction, rotate=False):
        value = self.manual_rotation_var.get() if rotate else self.manual_translation_var.get()
        action = np.zeros(7, dtype=np.float32)
        action[axis] = float(direction) * float(value)
        action[-1] = float(self.manual_gripper_var.get())
        self.manual_action(action)

    def manual_gripper_step(self, gripper):
        self.manual_gripper_var.set(float(gripper))
        action = np.zeros(7, dtype=np.float32)
        action[-1] = float(gripper)
        self.manual_action(action)

    def manual_action(self, action):
        if self.running:
            self.set_status("Pause before manual nudge buttons.")
            return
        if self.obs is None:
            self.reset_episode()

        print(f"manual_action={action}", flush=True)
        next_obs, _, _, truncated, _ = self.env.step(action)
        self.obs = next_obs
        if next_obs is not None and "full_image" in next_obs:
            self.update_image(next_obs["full_image"])

        self.step_idx += 1
        self.last_action = action
        self.action_var.set(f"manual_action={np.array2string(np.round(action, 3), precision=3, separator=', ')}")
        self.set_status(
            f"episode={self.episode_idx + 1} manual step={self.step_idx}/{self.cfg.max_steps} truncated={truncated}"
        )
        if truncated:
            self.finish_episode("environment truncated")

    def finish_episode(self, reason):
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        if not self.replay_images and self.step_idx == 0:
            self.set_status(f"No active episode to finish: {reason}")
            return

        if self.replay_images and self.args.save_video:
            save_rollout_video(self.replay_images, self.episode_idx)
        if self.replay_images and self.cfg.save_data:
            save_rollout_data(
                self.replay_images,
                self.rollout_images,
                self.rollout_states,
                self.rollout_actions,
                idx=self.episode_idx,
            )

        print(f"[bridge] episode {self.episode_idx + 1} finished: {reason}", flush=True)
        self.set_status(f"Episode {self.episode_idx + 1} finished: {reason}")
        self.episode_idx += 1
        if self.episode_idx >= self.cfg.max_episodes:
            self.set_status(f"Reached max_episodes={self.cfg.max_episodes}.")

    def close(self):
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.root.destroy()


def main():
    args = parse_args()

    global get_action, get_image_resize_size, get_model, get_processor, set_seed_everywhere
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_action, get_image_resize_size, get_model, set_seed_everywhere

    set_seed_everywhere(args.seed)
    cfg = make_cfg(args)

    global get_preprocessed_image, get_widowx_env, refresh_obs, save_rollout_data, save_rollout_video
    from experiments.robot.bridge.bridgev2_utils import (
        get_preprocessed_image,
        get_widowx_env,
        refresh_obs,
        save_rollout_data,
        save_rollout_video,
    )

    print(f"Checkpoint: {cfg.pretrained_checkpoint}")
    print(f"Bridge host: {cfg.host_ip}:{cfg.port}")
    print(f"Camera topic: {cfg.camera_topics[0]['name']}")

    model = get_model(cfg, DEVICE=args.device)
    model.eval()
    check_unnorm_key(cfg, model)
    processor = get_processor(cfg)
    env = get_widowx_env(cfg, model)

    root = tk.Tk()
    BridgeGui(root, args, cfg, model, processor, env)
    root.mainloop()


if __name__ == "__main__":
    main()
