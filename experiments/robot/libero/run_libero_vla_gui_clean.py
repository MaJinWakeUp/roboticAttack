"""
Run one clean LIBERO task in a visible MuJoCo viewer with OpenVLA control.

The user selects a LIBERO task prompt, then OpenVLA predicts actions from the
clean camera observation. No adversarial patch is loaded or applied.
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "LIBERO"))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    quat2axisangle,
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


CHECKPOINTS = {
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "libero_object": "openvla/openvla-7b-finetuned-libero-object",
    "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla/openvla-7b-finetuned-libero-10",
}

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
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
    return parser.parse_args()


def list_tasks(task_suite):
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        print(f"{task_id:2d}: {task.language}")


def choose_task_id(task_suite, task_id):
    if task_id is not None:
        if task_id < 0 or task_id >= task_suite.n_tasks:
            raise ValueError(f"task_id must be in [0, {task_suite.n_tasks - 1}]")
        return task_id

    list_tasks(task_suite)
    while True:
        choice = input("Select task id or search text> ").strip()
        if choice.isdigit():
            selected = int(choice)
            if 0 <= selected < task_suite.n_tasks:
                return selected
            print(f"task_id must be in [0, {task_suite.n_tasks - 1}]")
            continue

        needle = choice.lower()
        matches = [
            (idx, task_suite.get_task(idx).language)
            for idx in range(task_suite.n_tasks)
            if needle and needle in task_suite.get_task(idx).language.lower()
        ]
        if len(matches) == 1:
            print(f"Selected {matches[0][0]}: {matches[0][1]}")
            return matches[0][0]
        if matches:
            for idx, language in matches:
                print(f"{idx:2d}: {language}")
            print("Multiple matches; enter one task id.")
        else:
            print("No match; enter a task id or a substring from the prompt.")


def resolve_checkpoint(task_suite_name, checkpoint):
    if checkpoint != "auto":
        return checkpoint
    if task_suite_name not in CHECKPOINTS:
        raise ValueError(f"No default OpenVLA checkpoint for {task_suite_name}; pass --pretrained_checkpoint.")
    return CHECKPOINTS[task_suite_name]


def make_cfg(args, checkpoint):
    cfg = SimpleNamespace()
    cfg.model_family = "openvla"
    cfg.pretrained_checkpoint = checkpoint
    cfg.load_in_8bit = args.load_in_8bit
    cfg.load_in_4bit = args.load_in_4bit
    cfg.center_crop = args.center_crop
    cfg.task_suite_name = args.task_suite_name
    cfg.seed = args.seed
    cfg.unnorm_key = args.task_suite_name
    return cfg


def get_bddl_file(task):
    return str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)


def make_gui_env(args, task):
    env = ControlEnv(
        bddl_file_name=get_bddl_file(task),
        robots=["Panda"],
        controller="OSC_POSE",
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
        control_freq=20,
    )
    env.seed(args.seed)
    return env


def render(env):
    env.env.render()


def reset_env(env, task_suite, task_id, init_state_id):
    env.reset()
    try:
        initial_states = task_suite.get_task_init_states(task_id)
        if init_state_id < 0 or init_state_id >= len(initial_states):
            raise ValueError(f"init_state_id must be in [0, {len(initial_states) - 1}]")
        obs = env.set_init_state(initial_states[init_state_id])
    except Exception as exc:
        print(f"[warn] could not load fixed init state, using env.reset() state: {exc}")
        obs = env.reset()
    render(env)
    return obs


def check_unnorm_key(cfg, model):
    if not hasattr(model, "norm_stats"):
        return
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    if cfg.unnorm_key not in model.norm_stats:
        raise ValueError(f"Action un-norm key {cfg.unnorm_key} not found in VLA norm_stats.")


def run_policy(args, cfg, model, processor, env, task_suite, task_id, prompt):
    resize_size = get_image_resize_size(cfg)
    max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS.get(args.task_suite_name, 520)
    obs = reset_env(env, task_suite, task_id, args.init_state_id)

    for t in range(max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            render(env)
            continue

        img = get_libero_image(obs, resize_size)
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

    print(f"Task suite: {args.task_suite_name}")
    print(f"Task {task_id}: {task.language}")
    print(f"VLA prompt: {prompt}")
    print(f"Checkpoint: {checkpoint}")

    env = None
    try:
        model = get_model(cfg, DEVICE=args.device)
        model.eval()
        check_unnorm_key(cfg, model)
        processor = get_processor(cfg)
        env = make_gui_env(args, task)
        run_policy(args, cfg, model, processor, env, task_suite, task_id, prompt)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
