"""
Run one LIBERO task with a visible MuJoCo viewer and Tk button controls.

Hold a movement button to repeatedly step the robot. Release the button to stop.
Actions are OSC_POSE vectors:
    x y z roll pitch yaw gripper
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

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
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--robots", default="Panda")
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--action_scale", type=float, default=0.05)
    parser.add_argument("--rotation_scale", type=float, default=0.15)
    parser.add_argument("--gripper_hold", type=float, default=-1.0)
    parser.add_argument("--repeat_ms", type=int, default=80)
    return parser.parse_args()


def parse_robots(robots):
    names = [name.strip() for name in robots.split(",") if name.strip()]
    if not names:
        raise ValueError("--robots must include at least one robot name")
    return names


def get_bddl_file(task):
    return str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)


def make_action(axis=None, value=0.0, gripper_hold=-1.0):
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_hold], dtype=np.float32)
    if axis is not None:
        action[axis] = value
    return action


def render(env):
    env.env.render()


class ButtonController:
    def __init__(self, root, env, task_suite, task, args):
        self.root = root
        self.env = env
        self.task_suite = task_suite
        self.task = task
        self.args = args
        self.active_action = None
        self.active_after_id = None
        self.step_count = 0
        self.last_reward = 0.0
        self.last_done = False
        self.last_success = False

        self.status_var = tk.StringVar(value="Ready")
        self.scale_var = tk.DoubleVar(value=args.action_scale)
        self.rotation_var = tk.DoubleVar(value=args.rotation_scale)
        self.gripper_var = tk.DoubleVar(value=args.gripper_hold)

        self.build_ui()
        self.reset()

    def build_ui(self):
        self.root.title("LIBERO Manual Button Control")
        self.root.geometry("540x520")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(frame, text=f"{self.args.task_suite_name} / task {self.args.task_id}").grid(
            row=0, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(frame, text=self.task.language, wraplength=460).grid(row=1, column=0, columnspan=4, sticky="w")

        ttk.Separator(frame).grid(row=2, column=0, columnspan=4, pady=8, sticky="ew")

        self.add_hold_button(frame, "Forward +X", 3, 1, 0, 1)
        self.add_hold_button(frame, "Back -X", 5, 1, 0, -1)
        self.add_hold_button(frame, "Left +Y", 4, 0, 1, 1)
        self.add_hold_button(frame, "Right -Y", 4, 2, 1, -1)
        self.add_hold_button(frame, "Up +Z", 3, 3, 2, 1)
        self.add_hold_button(frame, "Down -Z", 5, 3, 2, -1)
        self.add_step_button(frame, "No-op", 4, 1, lambda: make_action(gripper_hold=self.gripper_var.get()))

        ttk.Separator(frame).grid(row=6, column=0, columnspan=4, pady=8, sticky="ew")

        self.add_hold_button(frame, "Roll +", 7, 0, 3, 1, rotate=True)
        self.add_hold_button(frame, "Roll -", 8, 0, 3, -1, rotate=True)
        self.add_hold_button(frame, "Pitch +", 7, 1, 4, 1, rotate=True)
        self.add_hold_button(frame, "Pitch -", 8, 1, 4, -1, rotate=True)
        self.add_hold_button(frame, "Yaw +", 7, 2, 5, 1, rotate=True)
        self.add_hold_button(frame, "Yaw -", 8, 2, 5, -1, rotate=True)

        self.add_hold_button(frame, "Open", 7, 3, 6, -1, gripper=True)
        self.add_hold_button(frame, "Close", 8, 3, 6, 1, gripper=True)

        ttk.Separator(frame).grid(row=9, column=0, columnspan=4, pady=8, sticky="ew")

        ttk.Label(frame, text="Move step").grid(row=10, column=0, sticky="w")
        ttk.Scale(frame, variable=self.scale_var, from_=0.005, to=0.15, orient="horizontal").grid(
            row=10, column=1, columnspan=3, sticky="ew"
        )
        ttk.Label(frame, text="Rotate step").grid(row=11, column=0, sticky="w")
        ttk.Scale(frame, variable=self.rotation_var, from_=0.01, to=0.5, orient="horizontal").grid(
            row=11, column=1, columnspan=3, sticky="ew"
        )
        ttk.Label(frame, text="Gripper hold").grid(row=12, column=0, sticky="w")
        ttk.Scale(frame, variable=self.gripper_var, from_=-1.0, to=1.0, orient="horizontal").grid(
            row=12, column=1, columnspan=3, sticky="ew"
        )

        ttk.Button(frame, text="Reset", command=self.reset).grid(row=13, column=0, pady=(10, 0), sticky="ew")
        ttk.Button(frame, text="Quit", command=self.close).grid(row=13, column=3, pady=(10, 0), sticky="ew")
        status_frame = ttk.Frame(frame, width=500, height=58)
        status_frame.grid(row=14, column=0, columnspan=4, pady=8, sticky="ew")
        status_frame.grid_propagate(False)
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=488, justify="left").grid(
            row=0, column=0, sticky="nw"
        )

        for column in range(4):
            frame.columnconfigure(column, weight=1)

        self.root.bind("<space>", lambda _event: self.step(make_action(gripper_hold=self.gripper_var.get())))
        self.root.bind("<Escape>", lambda _event: self.stop_action())

    def add_step_button(self, parent, text, row, column, action_fn):
        ttk.Button(parent, text=text, command=lambda: self.step(action_fn())).grid(
            row=row, column=column, padx=4, pady=4, sticky="ew"
        )

    def add_hold_button(self, parent, text, row, column, axis, direction, rotate=False, gripper=False):
        button = ttk.Button(parent, text=text)
        button.grid(row=row, column=column, padx=4, pady=4, sticky="ew")
        button.bind("<ButtonPress-1>", lambda _event: self.start_axis_action(axis, direction, rotate, gripper))
        button.bind("<ButtonRelease-1>", lambda _event: self.stop_action())

    def current_axis_action(self, axis, direction, rotate=False, gripper=False):
        if gripper:
            value = float(direction)
        elif rotate:
            value = float(direction) * float(self.rotation_var.get())
        else:
            value = float(direction) * float(self.scale_var.get())
        return make_action(axis, value, self.gripper_var.get())

    def start_axis_action(self, axis, direction, rotate=False, gripper=False):
        self.stop_action()
        self.active_action = lambda: self.current_axis_action(axis, direction, rotate, gripper)
        self.repeat_active_action()

    def repeat_active_action(self):
        if self.active_action is None:
            return
        self.step(self.active_action())
        self.active_after_id = self.root.after(self.args.repeat_ms, self.repeat_active_action)

    def stop_action(self):
        self.active_action = None
        if self.active_after_id is not None:
            self.root.after_cancel(self.active_after_id)
            self.active_after_id = None
        self.status_var.set("Stopped")

    def reset(self):
        self.stop_action()
        self.step_count = 0
        self.env.reset()
        try:
            initial_states = self.task_suite.get_task_init_states(self.args.task_id)
            obs = self.env.set_init_state(initial_states[self.args.init_state_id])
        except Exception as exc:
            self.status_var.set(f"Using env.reset() state; fixed init unavailable: {exc}")
            obs = self.env.reset()
        render(self.env)
        self.last_done = False
        self.last_success = False
        return obs

    def step(self, action):
        _, reward, done, _ = self.env.step(action.tolist())
        render(self.env)
        self.step_count += 1
        self.last_reward = float(reward)
        self.last_done = bool(done)
        self.last_success = bool(self.env.check_success())
        self.status_var.set(
            f"step={self.step_count} reward={self.last_reward:.3f} "
            f"done={self.last_done} success={self.last_success} "
            f"action={np.array2string(np.round(action, 3), precision=3, separator=', ', max_line_width=80)}"
        )
        if self.last_done or self.last_success:
            self.stop_action()

    def close(self):
        self.stop_action()
        self.env.close()
        self.root.destroy()


def main():
    args = parse_args()
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)

    env = ControlEnv(
        bddl_file_name=get_bddl_file(task),
        robots=parse_robots(args.robots),
        controller=args.controller,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=args.control_freq,
    )

    root = tk.Tk()
    ButtonController(root, env, task_suite, task, args)
    root.mainloop()


if __name__ == "__main__":
    main()
