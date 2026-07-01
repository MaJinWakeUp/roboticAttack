"""
Run one LIBERO task in a visible MuJoCo viewer with OpenVLA control under a
simulation-time adversarial image patch.

The simulator itself is clean. The patch is applied to the camera image that is
sent to OpenVLA before action prediction.
"""

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "LIBERO"))
sys.path.insert(0, str(REPO_ROOT / "VLAAttacker" / "white_patch"))

from appply_random_transform import RandomPatchTransform

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.libero.run_libero_vla_gui_clean import (
    MAX_STEPS,
    check_unnorm_key,
    choose_task_id,
    list_tasks,
    make_cfg,
    make_gui_env,
    render,
    reset_env,
    resolve_checkpoint,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from libero.libero import benchmark


DEFAULT_PATCH_POSITIONS = {
    "libero_10": (5, 160),
    "libero_object": (30, 150),
    "libero_goal": (15, 158),
    "libero_spatial": (120, 160),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--pretrained_checkpoint", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init_state_id", type=int, default=0)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--step_sleep", type=float, default=0.0)
    parser.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--list_tasks", action="store_true")
    parser.add_argument("--patchroot", required=True)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--shx", type=float, default=0.0)
    parser.add_argument("--shy", type=float, default=0.0)
    parser.add_argument("--patch_geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show_patch_view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch_view_scale", type=float, default=2.0)
    parser.add_argument("--patch_view_flip_x", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_patch_position(args):
    default_x, default_y = DEFAULT_PATCH_POSITIONS.get(args.task_suite_name, (5, 160))
    x = default_x if args.x is None else args.x
    y = default_y if args.y is None else args.y
    return x, y


def load_patch(path):
    patch_path = Path(path)
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file does not exist: {patch_path}")
    patch = torch.load(patch_path, map_location="cpu")
    if not isinstance(patch, torch.Tensor):
        raise TypeError(f"Expected patch tensor in {patch_path}, got {type(patch)}")
    return patch


def apply_patch(img, patch_transform, patch, args, position):
    return patch_transform.simulation_random_patch(
        img,
        patch,
        geometry=args.patch_geometry,
        colorjitter=False,
        angle=args.angle,
        shx=args.shx,
        shy=args.shy,
        position=position,
    )


class PatchViewer:
    def __init__(self, enabled, scale, flip_x):
        self.root = None
        self.image_label = None
        self.status_label = None
        self.photo = None
        self.scale = scale
        self.flip_x = flip_x
        if not enabled:
            return

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            print(f"[warn] could not open patch-view GUI: {exc}")
            return

        self.root.title("Patched OpenVLA Observation")
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=8, pady=(8, 4))
        self.status_label = tk.Label(self.root, text="Waiting for first patched observation")
        self.status_label.pack(padx=8, pady=(0, 8))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def update(self, img, step_idx):
        if self.root is None:
            return
        display_img = np.fliplr(img) if self.flip_x else img
        image = Image.fromarray(display_img).convert("RGB")
        if self.scale != 1.0:
            width = max(1, int(image.width * self.scale))
            height = max(1, int(image.height * self.scale))
            image = image.resize((width, height), Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)
        self.status_label.configure(text=f"Patched observation preview, step {step_idx}")
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        if self.root is None:
            return
        root = self.root
        self.root = None
        root.destroy()


def run_policy(args, cfg, model, processor, env, task_suite, task_id, prompt, patch, position, patch_viewer):
    resize_size = get_image_resize_size(cfg)
    patch_transform = RandomPatchTransform("cpu", False)
    max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS.get(args.task_suite_name, 520)
    obs = reset_env(env, task_suite, task_id, args.init_state_id)

    for t in range(max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            render(env)
            continue

        img = get_libero_image(obs, resize_size)
        img = apply_patch(img, patch_transform, patch, args, position)
        patch_viewer.update(img, t - args.num_steps_wait + 1)
        observation = {
            "full_image": img,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }

        action = get_action(cfg, model, observation, prompt, processor=processor, DEVICE=args.device)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)

        obs, reward, done, _ = env.step(action.tolist())
        render(env)
        success = env.check_success()
        print(f"step={t - args.num_steps_wait + 1} reward={reward:.3f} done={done} success={success}")
        if success or done:
            break
        if args.step_sleep > 0:
            time.sleep(args.step_sleep)


def main():
    args = parse_args()
    set_seed_everywhere(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    if args.list_tasks:
        list_tasks(task_suite)
        return

    task_id = choose_task_id(task_suite, args.task_id)
    task = task_suite.get_task(task_id)
    prompt = args.prompt or task.language
    checkpoint = resolve_checkpoint(args.task_suite_name, args.pretrained_checkpoint)
    cfg = make_cfg(args, checkpoint)
    patch = load_patch(args.patchroot)
    position = resolve_patch_position(args)

    print(f"Task suite: {args.task_suite_name}")
    print(f"Task {task_id}: {task.language}")
    print(f"VLA prompt: {prompt}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Patch: {args.patchroot}")
    print(f"Patch position: x={position[0]} y={position[1]} angle={args.angle} shx={args.shx} shy={args.shy}")

    env = None
    patch_viewer = PatchViewer(args.show_patch_view, args.patch_view_scale, args.patch_view_flip_x)
    try:
        model = get_model(cfg, DEVICE=args.device)
        model.eval()
        check_unnorm_key(cfg, model)
        processor = get_processor(cfg)
        env = make_gui_env(args, task)
        run_policy(args, cfg, model, processor, env, task_suite, task_id, prompt, patch, position, patch_viewer)
    finally:
        patch_viewer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
