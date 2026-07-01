"""
Run one LIBERO task with a visible MuJoCo viewer and manual stdin actions.

Actions are OSC_POSE vectors:
    x y z roll pitch yaw gripper
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "LIBERO"))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--init_state_id", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--robots", default="Panda")
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--action_scale", type=float, default=0.05)
    parser.add_argument("--rotation_scale", type=float, default=0.15)
    parser.add_argument("--gripper_hold", type=float, default=-1.0)
    parser.add_argument("--stop_on_done", action="store_true")
    return parser.parse_args()


def parse_robots(robots):
    if isinstance(robots, list):
        return robots
    names = [name.strip() for name in robots.split(",") if name.strip()]
    if not names:
        raise ValueError("--robots must include at least one robot name")
    return names


def render(env):
    env.env.render()


def get_bddl_file(task):
    return str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)


def reset_to_initial_state(env, task_suite, task_id, init_state_id):
    initial_states = task_suite.get_task_init_states(task_id)
    if init_state_id < 0 or init_state_id >= len(initial_states):
        raise ValueError(f"init_state_id must be in [0, {len(initial_states) - 1}]")
    env.reset()
    obs = env.set_init_state(initial_states[init_state_id])
    render(env)
    return obs


def named_action(name, action_scale, rotation_scale, gripper_hold):
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_hold], dtype=np.float32)
    aliases = {
        "noop": None,
        "x+": (0, action_scale),
        "x-": (0, -action_scale),
        "forward": (0, action_scale),
        "back": (0, -action_scale),
        "y+": (1, action_scale),
        "y-": (1, -action_scale),
        "left": (1, action_scale),
        "right": (1, -action_scale),
        "z+": (2, action_scale),
        "z-": (2, -action_scale),
        "up": (2, action_scale),
        "down": (2, -action_scale),
        "roll+": (3, rotation_scale),
        "roll-": (3, -rotation_scale),
        "pitch+": (4, rotation_scale),
        "pitch-": (4, -rotation_scale),
        "yaw+": (5, rotation_scale),
        "yaw-": (5, -rotation_scale),
        "open": (6, -1.0),
        "close": (6, 1.0),
    }
    if name not in aliases:
        return None
    update = aliases[name]
    if update is not None:
        action[update[0]] = update[1]
    return action


def parse_action(line, action_scale, rotation_scale, gripper_hold):
    cleaned = line.strip().lower().replace(",", " ")
    tokens = cleaned.split()
    if not tokens:
        return named_action("noop", action_scale, rotation_scale, gripper_hold)

    if len(tokens) == 1:
        action = named_action(tokens[0], action_scale, rotation_scale, gripper_hold)
        if action is not None:
            return action

    values = [float(token) for token in tokens]
    if len(values) == 6:
        values.append(gripper_hold)
    if len(values) != 7:
        raise ValueError("Enter a named command, 6 floats, or 7 floats.")
    return np.array(values, dtype=np.float32)


def print_help():
    print("\nCommands:")
    print("  q | quit | exit      stop")
    print("  reset                reset to the selected LIBERO initial state")
    print("  help                 show this message")
    print("  noop                 step with zero motion")
    print("  x+ x- y+ y- z+ z-    translate")
    print("  forward back left right up down")
    print("  roll+ roll- pitch+ pitch- yaw+ yaw-")
    print("  open | close         gripper only")
    print("  0 0 0 0 0 0 -1       raw 7-D action")
    print("  0 0 .05 0 0 0        raw 6-D action, gripper_hold appended\n")


def main():
    args = parse_args()
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    bddl_file = get_bddl_file(task)
    robots = parse_robots(args.robots)

    env = ControlEnv(
        bddl_file_name=bddl_file,
        robots=robots,
        controller=args.controller,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=args.control_freq,
    )

    print(f"Task suite: {args.task_suite_name}")
    print(f"Task {args.task_id}: {task.language}")
    print(f"BDDL: {bddl_file}")
    print_help()

    try:
        reset_to_initial_state(env, task_suite, args.task_id, args.init_state_id)
        for step_idx in range(args.max_steps):
            line = input(f"action[{step_idx}]> ").strip()
            if line.lower() in {"q", "quit", "exit"}:
                break
            if line.lower() in {"h", "help", "?"}:
                print_help()
                continue
            if line.lower() == "reset":
                reset_to_initial_state(env, task_suite, args.task_id, args.init_state_id)
                continue

            try:
                action = parse_action(line, args.action_scale, args.rotation_scale, args.gripper_hold)
            except ValueError as exc:
                print(f"Invalid action: {exc}")
                continue

            _, reward, done, _ = env.step(action.tolist())
            render(env)
            success = env.check_success()
            print(f"reward={reward:.3f} done={done} success={success} action={action.tolist()}")
            if success:
                print(f"[success] Task succeeded at manual step {step_idx + 1} (reward={reward:.3f}).", flush=True)
                break
            if done and args.stop_on_done:
                print(f"[done] Episode ended at manual step {step_idx + 1} before success (reward={reward:.3f}).", flush=True)
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
