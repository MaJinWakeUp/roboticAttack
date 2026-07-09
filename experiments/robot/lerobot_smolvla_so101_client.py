#!/usr/bin/env python3
"""Run an SO-101 follower arm from a remote SmolVLA inference server.

The client runs beside the robot. It reads the six SO-101 joint positions,
captures one to three OpenCV cameras, sends them to ``smolvla_server.py``, and
optionally sends the returned six joint targets to the follower arm. Execution
is opt-in; start with the default observe-only mode.

The default camera mapping matches ``Sa74ll/smolvla_so101_pickandplace``:
``camera1`` is the training ``up`` camera and ``camera2`` is the training
``side`` camera. A third camera is optional.
"""

import argparse
import base64
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


LOGGER = logging.getLogger("lerobot_smolvla_so101_client")
SO101_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server_url", default="http://127.0.0.1:8000")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--task", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--session_id", default="")

    parser.add_argument("--camera1_index", required=True, help="Training 'up' camera index or path.")
    parser.add_argument("--camera2_index", default="", help="Optional training 'side' camera index or path.")
    parser.add_argument("--camera3_index", default="", help="Optional third camera index or path.")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=float, default=30.0)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--warmup_frames", type=int, default=5)
    parser.add_argument("--preview", action="store_true")

    parser.add_argument("--robot_port", required=True)
    parser.add_argument("--robot_id", default="smolvla_so101_client")
    parser.add_argument(
        "--max_relative_target",
        type=int,
        default=10,
        help="Maximum per-step target change enforced by LeRobot; use a conservative value while testing.",
    )
    parser.add_argument("--keep_torque_on_disconnect", dest="disable_torque_on_disconnect", action="store_false")
    parser.add_argument("--no_calibrate", dest="calibrate", action="store_false")
    parser.add_argument("--execute", action="store_true", help="Actually send returned joint targets to the arm.")
    parser.add_argument("--no_confirm", dest="confirm", action="store_false")
    parser.add_argument("--max_action_abs", type=float, default=None)

    parser.add_argument("--steps", type=int, default=0, help="0 means run until Ctrl-C.")
    parser.add_argument("--control_hz", type=float, default=30.0)
    parser.add_argument(
        "--no_action_chunk",
        dest="use_action_chunk",
        action="store_false",
        help="Request one action per HTTP call instead of the policy's native action chunks.",
    )
    parser.add_argument("--save_jsonl", default="")
    parser.add_argument("--log_level", default="INFO")

    parser.add_argument("--ssh_tunnel_host", default="", help="SSH host that can reach the inference server.")
    parser.add_argument("--ssh_tunnel_user", default="")
    parser.add_argument("--ssh_tunnel_port", type=int, default=22)
    parser.add_argument("--ssh_tunnel_key", default="")
    parser.add_argument("--ssh_tunnel_jump", default="")
    parser.add_argument("--ssh_local_host", default="127.0.0.1")
    parser.add_argument("--ssh_local_port", type=int, default=18000)
    parser.add_argument("--ssh_remote_host", default="127.0.0.1")
    parser.add_argument("--ssh_remote_port", type=int, default=8000)
    parser.add_argument("--ssh_ready_timeout", type=float, default=20.0)
    parser.set_defaults(calibrate=True, confirm=True, disable_torque_on_disconnect=True, use_action_chunk=True)
    return parser.parse_args()


def parse_camera_index(value: str) -> Union[int, str]:
    return int(value) if value.isdigit() else value


class OpenCVCamera:
    def __init__(self, name: str, index_or_path: Union[int, str], args: argparse.Namespace):
        self.name = name
        self.index_or_path = index_or_path
        self.width = args.camera_width
        self.height = args.camera_height
        self.fps = args.camera_fps
        self.jpeg_quality = int(np.clip(args.jpeg_quality, 1, 100))
        self.cv2 = None
        self.capture = None

    def connect(self, warmup_frames: int) -> None:
        import cv2

        self.cv2 = cv2
        cv2.setNumThreads(1)
        self.capture = cv2.VideoCapture(self.index_or_path)
        if not self.capture.isOpened():
            raise ConnectionError(f"Could not open {self.name} OpenCV camera {self.index_or_path!r}.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.capture.set(cv2.CAP_PROP_FPS, float(self.fps))
        for _ in range(max(0, warmup_frames)):
            self.capture.read()
        LOGGER.info("Connected %s camera %r.", self.name, self.index_or_path)

    def read_jpeg(self) -> Tuple[bytes, np.ndarray]:
        if self.capture is None or self.cv2 is None:
            raise RuntimeError(f"{self.name} camera is not connected.")
        ok, frame_bgr = self.capture.read()
        if not ok:
            raise RuntimeError(f"{self.name} OpenCV camera read failed.")
        ok, encoded = self.cv2.imencode(
            ".jpg", frame_bgr, [int(self.cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError(f"{self.name} JPEG encoding failed.")
        return encoded.tobytes(), frame_bgr

    def disconnect(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def make_cameras(args: argparse.Namespace) -> Dict[str, OpenCVCamera]:
    requested = {
        "camera1": args.camera1_index,
        "camera2": args.camera2_index,
        "camera3": args.camera3_index,
    }
    return {
        name: OpenCVCamera(name, parse_camera_index(value), args)
        for name, value in requested.items()
        if value
    }


def build_ssh_destination(args: argparse.Namespace) -> str:
    if not args.ssh_tunnel_host:
        raise ValueError("--ssh_tunnel_host is required to start an SSH tunnel.")
    return f"{args.ssh_tunnel_user}@{args.ssh_tunnel_host}" if args.ssh_tunnel_user else args.ssh_tunnel_host


def wait_for_local_port(host: str, port: int, timeout_s: float, process: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    last_error: Optional[OSError] = None
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


def start_ssh_tunnel(args: argparse.Namespace) -> Optional[subprocess.Popen]:
    if not args.ssh_tunnel_host:
        return None
    command = [
        "ssh",
        "-N",
        "-L",
        f"{args.ssh_local_host}:{args.ssh_local_port}:{args.ssh_remote_host}:{args.ssh_remote_port}",
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
    return process


def stop_ssh_tunnel(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def connect_robot(args: argparse.Namespace) -> Any:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=args.disable_torque_on_disconnect,
    )
    robot = SO101Follower(config)
    try:
        robot.connect(calibrate=args.calibrate)
    except TypeError:
        robot.connect()
    LOGGER.info("Connected SO-101 follower %s.", robot)
    return robot


def state_from_observation(observation: Dict[str, Any]) -> List[float]:
    missing = [key for key in SO101_JOINT_KEYS if key not in observation]
    if missing:
        raise KeyError(f"SO-101 observation is missing joint keys: {', '.join(missing)}")
    state = [float(observation[key]) for key in SO101_JOINT_KEYS]
    if not np.isfinite(state).all():
        raise ValueError("Robot observation contains a non-finite joint position.")
    return state


def post_predict(
    args: argparse.Namespace,
    state: List[float],
    images: Dict[str, bytes],
    request_id: str,
    session_id: str,
    return_action_chunk: bool,
) -> Dict[str, Any]:
    payload = {
        "request_id": request_id,
        "session_id": session_id,
        "task": args.task,
        "state": state,
        "images": {key: base64.b64encode(value).decode("ascii") for key, value in images.items()},
        "return_action_chunk": return_action_chunk,
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    request = urllib.request.Request(
        args.server_url.rstrip("/") + "/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Server returned HTTP {exc.code}: {error_body}") from exc


def maybe_confirm_execution(args: argparse.Namespace) -> None:
    if not args.execute or not args.confirm:
        return
    print("\nAbout to send SmolVLA joint targets to the SO-101 follower arm.")
    print("Check that the arm is clear and that its calibration matches the model's training setup.")
    input("Press Enter to start, or Ctrl-C to abort...")


def show_preview(cameras: Dict[str, OpenCVCamera], frame: np.ndarray, action: np.ndarray) -> bool:
    camera = cameras["camera1"]
    if camera.cv2 is None:
        return True
    label = np.array2string(np.round(action, 2), precision=2, separator=", ")
    camera.cv2.putText(frame, label[:110], (10, 28), camera.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    camera.cv2.imshow("SmolVLA SO-101 camera1", frame)
    return camera.cv2.waitKey(1) & 0xFF != ord("q")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s: %(message)s")
    cameras = make_cameras(args)
    robot = None
    tunnel_process = None
    log_f = None
    session_id = args.session_id or str(uuid.uuid4())
    try:
        tunnel_process = start_ssh_tunnel(args)
        robot = connect_robot(args)
        for camera in cameras.values():
            camera.connect(args.warmup_frames)
        maybe_confirm_execution(args)
        if args.save_jsonl:
            log_f = Path(args.save_jsonl).open("a")

        period_s = 1.0 / args.control_hz if args.control_hz > 0 else 0.0
        step = 0
        pending_actions: List[np.ndarray] = []
        while args.steps == 0 or step < args.steps:
            started = time.time()
            observation = robot.get_observation()
            state = state_from_observation(observation)
            camera1_frame = None
            image_bytes: Dict[str, bytes] = {}
            response: Dict[str, Any] = {}
            request_id = ""
            if not pending_actions:
                for name, camera in cameras.items():
                    encoded, frame = camera.read_jpeg()
                    image_bytes[name] = encoded
                    if name == "camera1":
                        camera1_frame = frame
                request_id = str(uuid.uuid4())
                response = post_predict(
                    args,
                    state,
                    image_bytes,
                    request_id,
                    session_id,
                    args.use_action_chunk,
                )
                if not response.get("ok", False):
                    raise RuntimeError(f"Prediction failed: {response}")
                if args.use_action_chunk:
                    action_chunk = np.asarray(response.get("action_chunk"), dtype=np.float32)
                    if action_chunk.ndim != 2 or action_chunk.shape[1] != len(SO101_JOINT_KEYS):
                        raise ValueError(
                            "Server did not return a [steps, 6] action_chunk for the SO-101 policy."
                        )
                    pending_actions = [row for row in action_chunk]
            if pending_actions:
                action = pending_actions.pop(0)
            else:
                action = np.asarray(response["action"], dtype=np.float32)
            if action.size != len(SO101_JOINT_KEYS):
                raise ValueError(f"Expected {len(SO101_JOINT_KEYS)} SmolVLA action values, got {action.size}.")
            if args.max_action_abs is not None:
                action = np.clip(action, -args.max_action_abs, args.max_action_abs)
            mapped_action = dict(zip(SO101_JOINT_KEYS, action.tolist(), strict=True))
            sent_action = robot.send_action(mapped_action) if args.execute else None

            elapsed = time.time() - started
            print(
                f"step={step:04d} latency={response.get('latency_s', 0.0):.3f}s "
                f"loop={elapsed:.3f}s state={np.round(state, 2).tolist()} "
                f"action={np.round(action, 2).tolist()} pending={len(pending_actions)} sent={sent_action}",
                flush=True,
            )
            if log_f is not None:
                log_f.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "step": step,
                            "request_id": request_id,
                            "session_id": session_id,
                            "task": args.task,
                            "state": state,
                            "image_keys": sorted(image_bytes),
                            "action": action.tolist(),
                            "sent_action": sent_action,
                            "server": response,
                        }
                    )
                    + "\n"
                )
                log_f.flush()
            if args.preview and camera1_frame is not None and not show_preview(cameras, camera1_frame, action):
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
        for camera in cameras.values():
            camera.disconnect()
        if cameras and next(iter(cameras.values())).cv2 is not None:
            next(iter(cameras.values())).cv2.destroyAllWindows()
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                LOGGER.warning("Robot disconnect failed.", exc_info=True)
        stop_ssh_tunnel(tunnel_process)


if __name__ == "__main__":
    main()
