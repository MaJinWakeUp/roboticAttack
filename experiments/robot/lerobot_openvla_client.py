#!/usr/bin/env python3
"""
OpenCV + LeRobot client for a remote OpenVLA inference server.

The client captures frames locally, asks the GPU server for OpenVLA actions, and
optionally sends mapped actions to a LeRobot robot. Robot execution is opt-in:
use --execute only after checking the action mapping for your hardware.
"""

import argparse
import base64
import importlib
import json
import logging
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

LOGGER = logging.getLogger("lerobot_openvla_client")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server_url", default="http://127.0.0.1:8000")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--task", required=True)
    parser.add_argument("--unnorm_key", default="")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)

    parser.add_argument("--camera_index", default="0", help="OpenCV camera index or path, e.g. 0 or /dev/video0.")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=float, default=30.0)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--warmup_frames", type=int, default=5)
    parser.add_argument("--preview", action="store_true")

    parser.add_argument("--steps", type=int, default=0, help="0 means run until Ctrl-C.")
    parser.add_argument("--control_hz", type=float, default=5.0)
    parser.add_argument("--save_jsonl", default="")
    parser.add_argument("--log_level", default="INFO")

    parser.add_argument("--ssh_tunnel_host", default="", help="SSH host that can reach the OpenVLA server.")
    parser.add_argument("--ssh_tunnel_user", default="")
    parser.add_argument("--ssh_tunnel_port", type=int, default=22)
    parser.add_argument("--ssh_tunnel_key", default="")
    parser.add_argument("--ssh_tunnel_jump", default="", help="Optional ProxyJump host, e.g. user@login.cluster.edu.")
    parser.add_argument("--ssh_local_host", default="127.0.0.1")
    parser.add_argument("--ssh_local_port", type=int, default=18000)
    parser.add_argument("--ssh_remote_host", default="127.0.0.1")
    parser.add_argument("--ssh_remote_port", type=int, default=8000)
    parser.add_argument("--ssh_ready_timeout", type=float, default=20.0)

    parser.add_argument(
        "--robot_type",
        choices=["none", "so100_follower", "so100_follower_end_effector", "so101_follower", "custom"],
        default="none",
    )
    parser.add_argument("--robot_port", default="")
    parser.add_argument("--robot_id", default="openvla_client")
    parser.add_argument("--max_relative_target", type=int, default=None)
    parser.add_argument("--keep_torque_on_disconnect", dest="disable_torque_on_disconnect", action="store_false")
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    parser.add_argument("--execute", action="store_true", help="Actually call robot.send_action(...).")
    parser.add_argument("--no-confirm", dest="confirm", action="store_false")

    parser.add_argument("--urdf_path", default="", help="Required for so100_follower_end_effector.")
    parser.add_argument("--target_frame_name", default="gripper_frame_link")
    parser.add_argument("--max_gripper_pos", type=float, default=50.0)

    parser.add_argument("--robot_class_path", default="", help="For custom robots, e.g. package.module:RobotClass.")
    parser.add_argument("--robot_config_class_path", default="", help="For custom robots, e.g. package.module:ConfigClass.")
    parser.add_argument("--robot_config_json", default="", help="Inline JSON object for the custom robot config.")
    parser.add_argument("--robot_config_path", default="", help="Path to a JSON object for the custom robot config.")

    parser.add_argument(
        "--action_keys",
        default="",
        help="Comma-separated robot action keys. Required for most robots when --execute is set.",
    )
    parser.add_argument(
        "--action_indexes",
        default="",
        help="Comma-separated OpenVLA action indexes for action_keys. Defaults to 0..N-1.",
    )
    parser.add_argument("--action_scales", default="1.0", help="One scalar or one comma-separated scalar per key.")
    parser.add_argument("--action_offsets", default="0.0", help="One scalar or one comma-separated scalar per key.")
    parser.add_argument(
        "--gripper_transform",
        choices=["auto", "none", "openvla_to_so100_ee", "threshold_to_so100_ee"],
        default="auto",
    )
    parser.add_argument("--gripper_threshold", type=float, default=0.5)
    parser.add_argument("--max_action_abs", type=float, default=None)
    parser.set_defaults(calibrate=True, confirm=True, disable_torque_on_disconnect=True)
    return parser.parse_args()


def parse_camera_index(value: str) -> Union[int, str]:
    return int(value) if value.isdigit() else value


def parse_csv(value: str, cast: Any = float) -> List[Any]:
    if not value:
        return []
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def expand(values: List[float], n: int, name: str) -> List[float]:
    if not values:
        return [0.0] * n
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{name} must contain one value or exactly {n} values.")
    return values


def import_symbol(path: str) -> Any:
    if ":" in path:
        module_name, symbol_name = path.split(":", 1)
    else:
        module_name, symbol_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def build_ssh_destination(args: argparse.Namespace) -> str:
    if not args.ssh_tunnel_host:
        raise ValueError("--ssh_tunnel_host is required to start an SSH tunnel.")
    if args.ssh_tunnel_user:
        return f"{args.ssh_tunnel_user}@{args.ssh_tunnel_host}"
    return args.ssh_tunnel_host


def start_ssh_tunnel(args: argparse.Namespace) -> Optional[subprocess.Popen]:
    if not args.ssh_tunnel_host:
        return None

    local_forward = (
        f"{args.ssh_local_host}:{args.ssh_local_port}:"
        f"{args.ssh_remote_host}:{args.ssh_remote_port}"
    )
    command = [
        "ssh",
        "-N",
        "-L",
        local_forward,
        "-p",
        str(args.ssh_tunnel_port),
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if args.ssh_tunnel_key:
        command.extend(["-i", args.ssh_tunnel_key])
    if args.ssh_tunnel_jump:
        command.extend(["-J", args.ssh_tunnel_jump])
    command.append(build_ssh_destination(args))

    LOGGER.info("Starting SSH tunnel: %s", " ".join(command))
    process = subprocess.Popen(command)
    wait_for_local_port(args.ssh_local_host, args.ssh_local_port, args.ssh_ready_timeout, process)
    args.server_url = f"http://{args.ssh_local_host}:{args.ssh_local_port}"
    LOGGER.info(
        "SSH tunnel ready. Client will use %s -> %s:%s on %s.",
        args.server_url,
        args.ssh_remote_host,
        args.ssh_remote_port,
        build_ssh_destination(args),
    )
    return process


def wait_for_local_port(host: str, port: int, timeout_s: float, process: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SSH tunnel exited early with status {process.returncode}.")
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for SSH tunnel on {host}:{port}: {last_error}")


def stop_ssh_tunnel(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def load_json_object(inline_json: str, path: str) -> Dict[str, Any]:
    if inline_json and path:
        raise ValueError("Use only one of --robot_config_json and --robot_config_path.")
    if inline_json:
        data = json.loads(inline_json)
    elif path:
        with Path(path).open("r") as f:
            data = json.load(f)
    else:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Robot config must be a JSON object.")
    return data


class OpenCVCamera:
    def __init__(self, index_or_path: Union[int, str], width: int, height: int, fps: float, jpeg_quality: int):
        self.index_or_path = index_or_path
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.cv2 = None
        self.capture = None

    def connect(self, warmup_frames: int) -> None:
        import cv2

        self.cv2 = cv2
        cv2.setNumThreads(1)
        self.capture = cv2.VideoCapture(self.index_or_path)
        if not self.capture.isOpened():
            raise ConnectionError(f"Could not open OpenCV camera {self.index_or_path!r}.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.capture.set(cv2.CAP_PROP_FPS, float(self.fps))
        for _ in range(max(0, warmup_frames)):
            ok, _ = self.capture.read()
            if not ok:
                time.sleep(0.05)
        LOGGER.info("Connected OpenCV camera %r.", self.index_or_path)

    def read_jpeg(self) -> Tuple[bytes, np.ndarray]:
        if self.capture is None or self.cv2 is None:
            raise RuntimeError("Camera is not connected.")
        ok, frame_bgr = self.capture.read()
        if not ok:
            raise RuntimeError("OpenCV camera read failed.")
        encode_params = [int(self.cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, encoded = self.cv2.imencode(".jpg", frame_bgr, encode_params)
        if not ok:
            raise RuntimeError("JPEG encoding failed.")
        return encoded.tobytes(), frame_bgr

    def show(self, frame_bgr: np.ndarray, action: Optional[np.ndarray] = None) -> bool:
        if self.cv2 is None:
            return True
        if action is not None:
            label = np.array2string(np.round(action, 3), precision=3, separator=", ")
            self.cv2.putText(
                frame_bgr,
                label[:110],
                (10, 28),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                self.cv2.LINE_AA,
            )
        self.cv2.imshow("OpenVLA client camera", frame_bgr)
        return self.cv2.waitKey(1) & 0xFF != ord("q")

    def disconnect(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.cv2 is not None:
            self.cv2.destroyAllWindows()


def post_predict(args: argparse.Namespace, image_bytes: bytes, request_id: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "request_id": request_id,
        "task": args.task,
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
        "image_format": "jpeg",
    }
    if args.unnorm_key:
        payload["unnorm_key"] = args.unnorm_key
    if args.center_crop:
        payload["center_crop"] = True

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    url = args.server_url.rstrip("/") + "/predict"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Server returned HTTP {exc.code}: {error_body}") from exc


def make_builtin_robot(args: argparse.Namespace) -> Any:
    if args.robot_type in {"so100_follower", "so100_follower_end_effector"}:
        if not args.robot_port:
            raise ValueError(f"--robot_port is required for {args.robot_type}.")
        from lerobot.robots.so100_follower import (
            SO100Follower,
            SO100FollowerConfig,
            SO100FollowerEndEffector,
            SO100FollowerEndEffectorConfig,
        )

        common = {
            "port": args.robot_port,
            "id": args.robot_id,
            "max_relative_target": args.max_relative_target,
            "disable_torque_on_disconnect": args.disable_torque_on_disconnect,
        }
        if args.robot_type == "so100_follower":
            return SO100Follower(SO100FollowerConfig(**common))
        if not args.urdf_path:
            raise ValueError("--urdf_path is required for so100_follower_end_effector.")
        config = SO100FollowerEndEffectorConfig(
            **common,
            urdf_path=args.urdf_path,
            target_frame_name=args.target_frame_name,
            max_gripper_pos=args.max_gripper_pos,
        )
        return SO100FollowerEndEffector(config)

    if args.robot_type == "so101_follower":
        if not args.robot_port:
            raise ValueError("--robot_port is required for so101_follower.")
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        config = SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            max_relative_target=args.max_relative_target,
            disable_torque_on_disconnect=args.disable_torque_on_disconnect,
        )
        return SO101Follower(config)

    raise ValueError(f"Unsupported built-in robot type: {args.robot_type}")


def make_custom_robot(args: argparse.Namespace) -> Any:
    if not args.robot_class_path:
        raise ValueError("--robot_class_path is required when --robot_type custom.")
    robot_cls = import_symbol(args.robot_class_path)
    config_kwargs = load_json_object(args.robot_config_json, args.robot_config_path)
    if args.robot_port and "port" not in config_kwargs:
        config_kwargs["port"] = args.robot_port
    if args.robot_id and "id" not in config_kwargs:
        config_kwargs["id"] = args.robot_id

    if args.robot_config_class_path:
        config_cls = import_symbol(args.robot_config_class_path)
        return robot_cls(config_cls(**config_kwargs))
    return robot_cls(config_kwargs)


def connect_robot(args: argparse.Namespace) -> Optional[Any]:
    if args.robot_type == "none":
        if args.execute:
            raise ValueError("--execute requires --robot_type other than none.")
        return None

    robot = make_custom_robot(args) if args.robot_type == "custom" else make_builtin_robot(args)
    try:
        robot.connect(calibrate=args.calibrate)
    except TypeError:
        robot.connect()
    LOGGER.info("Connected robot: %s", robot)
    return robot


def default_action_mapping(args: argparse.Namespace) -> Tuple[List[str], List[int], str]:
    if args.action_keys:
        keys = parse_csv(args.action_keys, str)
        indexes = parse_csv(args.action_indexes, int) if args.action_indexes else list(range(len(keys)))
        return keys, indexes, args.gripper_transform

    if args.robot_type == "so100_follower_end_effector":
        transform = "openvla_to_so100_ee" if args.gripper_transform == "auto" else args.gripper_transform
        return ["delta_x", "delta_y", "delta_z", "gripper"], [0, 1, 2, 6], transform

    raise ValueError(
        "No safe default action mapping for this robot. Provide --action_keys and, if needed, --action_indexes."
    )


def transform_gripper(value: float, transform: str, threshold: float) -> float:
    if transform in {"auto", "none"}:
        return value
    if transform == "openvla_to_so100_ee":
        return float(np.clip(value, 0.0, 1.0) * 2.0)
    if transform == "threshold_to_so100_ee":
        return 2.0 if value >= threshold else 0.0
    raise ValueError(f"Unknown gripper transform: {transform}")


def map_action_for_robot(args: argparse.Namespace, action: np.ndarray) -> Dict[str, float]:
    keys, indexes, gripper_transform = default_action_mapping(args)
    if len(keys) != len(indexes):
        raise ValueError("--action_keys and --action_indexes must have the same length.")
    if max(indexes) >= len(action) or min(indexes) < 0:
        raise ValueError(f"Action indexes {indexes} are outside returned action dimension {len(action)}.")

    scales = expand(parse_csv(args.action_scales, float), len(keys), "--action_scales")
    offsets = expand(parse_csv(args.action_offsets, float), len(keys), "--action_offsets")

    mapped: Dict[str, float] = {}
    for key, index, scale, offset in zip(keys, indexes, scales, offsets):
        value = float(action[index])
        if key == "gripper":
            value = transform_gripper(value, gripper_transform, args.gripper_threshold)
        mapped[key] = float(value * scale + offset)
    return mapped


def maybe_confirm_execution(args: argparse.Namespace) -> None:
    if not args.execute or not args.confirm:
        return
    print("\nAbout to send mapped OpenVLA actions to the robot.")
    print("Check that the robot is clear, enabled, and that the action mapping is correct.")
    input("Press Enter to start, or Ctrl-C to abort...")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s: %(message)s")

    camera = OpenCVCamera(
        index_or_path=parse_camera_index(args.camera_index),
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        jpeg_quality=args.jpeg_quality,
    )
    robot = None
    log_f = None
    tunnel_process = None
    try:
        tunnel_process = start_ssh_tunnel(args)
        robot = connect_robot(args)
        camera.connect(args.warmup_frames)
        maybe_confirm_execution(args)

        if args.save_jsonl:
            log_f = Path(args.save_jsonl).open("a")

        period_s = 1.0 / args.control_hz if args.control_hz > 0 else 0.0
        step = 0
        while args.steps == 0 or step < args.steps:
            started = time.time()
            request_id = str(uuid.uuid4())
            image_bytes, frame_bgr = camera.read_jpeg()
            response = post_predict(args, image_bytes=image_bytes, request_id=request_id)
            if not response.get("ok", False):
                raise RuntimeError(f"Prediction failed: {response}")

            action = np.asarray(response["action"], dtype=np.float32)
            if args.max_action_abs is not None:
                action = np.clip(action, -args.max_action_abs, args.max_action_abs)

            mapped_action = None
            sent_action = None
            if args.execute:
                if robot is None:
                    raise RuntimeError("No robot connected.")
                mapped_action = map_action_for_robot(args, action)
                sent_action = robot.send_action(mapped_action)

            elapsed = time.time() - started
            print(
                f"step={step:04d} latency={response.get('latency_s', 0.0):.3f}s "
                f"loop={elapsed:.3f}s action={np.round(action, 4).tolist()} "
                f"mapped={mapped_action}",
                flush=True,
            )

            if log_f is not None:
                log_f.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "step": step,
                            "request_id": request_id,
                            "task": args.task,
                            "action": action.tolist(),
                            "mapped_action": mapped_action,
                            "sent_action": sent_action,
                            "server": response,
                        }
                    )
                    + "\n"
                )
                log_f.flush()

            if args.preview and not camera.show(frame_bgr, action):
                break

            step += 1
            sleep_s = period_s - (time.time() - started)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
    finally:
        if log_f is not None:
            log_f.close()
        camera.disconnect()
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                LOGGER.warning("Robot disconnect failed.", exc_info=True)
        stop_ssh_tunnel(tunnel_process)


if __name__ == "__main__":
    main()
