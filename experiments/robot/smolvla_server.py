#!/usr/bin/env python3
"""Serve a LeRobot SmolVLA policy through a small JSON/HTTP API.

Run this on the GPU server. ``POST /predict`` accepts an SO-101 observation and
returns one continuous action from the policy's action chunk. SmolVLA requires
the current robot state; it is deliberately not replaced with zeros because
that produces misleading and potentially unsafe robot commands.

Request body::

    {
      "task": "put the block in the box",
      "state": [six SO-101 joint positions],
      "images": {
        "camera1": "<base64 JPEG>",
        "camera2": "<base64 JPEG>",
        "camera3": "<base64 JPEG>"
      },
      "session_id": "robot-session-1"
    }

For a single camera, ``image_b64`` is accepted and is mapped to ``camera1`` by
default. The default SO-101 checkpoint's published inference example uses two
views (``camera1`` and ``camera2``), so provide both whenever possible.
"""

import argparse
import base64
import io
import json
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


LOGGER = logging.getLogger("smolvla_server")
OBS_STATE = "observation.state"
OBS_IMAGES_PREFIX = "observation.images."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("SMOLVLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SMOLVLA_PORT", "8000")))
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CHECKPOINT", "Sa74ll/smolvla_so101_pickandplace"),
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument(
        "--single_image_key",
        default=os.environ.get("SINGLE_IMAGE_KEY", "observation.images.camera1"),
        help="Configured image feature used when a request supplies image_b64 instead of images.",
    )
    parser.add_argument("--api_key", default=os.environ.get("SMOLVLA_API_KEY", ""))
    parser.add_argument("--max_request_bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--log_level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def load_policy(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    """Load the policy and its checkpoint-specific pre/post-processors lazily."""
    try:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as exc:
        raise ImportError(
            "SmolVLA requires LeRobot with its smolvla extra. In the server environment run "
            "`pip install -e '/path/to/lerobot[smolvla]'` (or `pip install 'lerobot[smolvla]'`)."
        ) from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")

    LOGGER.info("Loading SmolVLA policy from %s", args.checkpoint)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    return policy, preprocess, postprocess, torch


def feature_shape(feature: Any) -> Sequence[int]:
    if hasattr(feature, "shape"):
        return tuple(int(x) for x in feature.shape)
    if isinstance(feature, Mapping) and "shape" in feature:
        return tuple(int(x) for x in feature["shape"])
    raise ValueError(f"Cannot determine feature shape from {feature!r}")


def decode_image(image_b64: str) -> np.ndarray:
    from PIL import Image

    if image_b64.lstrip().startswith("data:") and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_b64, validate=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb).copy()


def image_to_tensor(image: np.ndarray, torch_module: Any) -> Any:
    """Convert HWC uint8 RGB to LeRobot's unbatched CHW float image format."""
    return torch_module.from_numpy(image).permute(2, 0, 1).contiguous().float().div_(255.0)


def action_to_list(action: Any) -> List[float]:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    return [float(value) for value in values]


def action_chunk_to_list(action_chunk: Any) -> List[List[float]]:
    if hasattr(action_chunk, "detach"):
        action_chunk = action_chunk.detach().cpu().numpy()
    values = np.asarray(action_chunk, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected an action chunk with shape [steps, action_dim], got {values.shape}.")
    return [[float(value) for value in row] for row in values]


@dataclass
class SmolVLARuntime:
    policy: Any
    preprocess: Any
    postprocess: Any
    torch_module: Any
    checkpoint: str
    device: str
    state_dim: int
    image_keys: List[str]
    single_image_key: str
    lock: threading.Lock
    active_session_id: Optional[str] = None

    def resolve_image_key(self, key: str) -> str:
        full_key = key if key.startswith(OBS_IMAGES_PREFIX) else OBS_IMAGES_PREFIX + key
        if full_key not in self.image_keys:
            expected = ", ".join(self.image_keys)
            raise ValueError(f"Unknown image key {key!r}; expected one of: {expected}")
        return full_key

    def make_observation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task = str(payload.get("task", "")).strip()
        if not task:
            raise ValueError("missing_task")

        raw_state = payload.get("state")
        if not isinstance(raw_state, list) or len(raw_state) != self.state_dim:
            raise ValueError(f"state must be a JSON list of exactly {self.state_dim} values.")
        state = np.asarray(raw_state, dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("state contains a non-finite value.")

        raw_images = payload.get("images")
        decoded_images: Dict[str, np.ndarray] = {}
        if raw_images is not None:
            if not isinstance(raw_images, Mapping):
                raise ValueError("images must be a JSON object mapping camera keys to base64 images.")
            for key, encoded in raw_images.items():
                if not isinstance(encoded, str):
                    raise ValueError(f"Image for {key!r} must be a base64 string.")
                decoded_images[self.resolve_image_key(str(key))] = decode_image(encoded)
        else:
            encoded = payload.get("image_b64") or payload.get("image")
            if not encoded:
                raise ValueError("missing_images: provide images or image_b64.")
            decoded_images[self.resolve_image_key(str(payload.get("image_key", self.single_image_key)))] = decode_image(
                str(encoded)
            )

        observation: Dict[str, Any] = {
            OBS_STATE: self.torch_module.from_numpy(state),
            "task": task,
        }
        for key, image in decoded_images.items():
            observation[key] = image_to_tensor(image, self.torch_module)
        return observation

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        observation = self.make_observation(payload)
        session_id = str(payload.get("session_id", "default"))
        reset = bool(payload.get("reset", False))

        with self.lock, self.torch_module.inference_mode():
            if reset or session_id != self.active_session_id:
                self.policy.reset()
                self.active_session_id = session_id
            batch = self.preprocess(observation)
            if bool(payload.get("return_action_chunk", False)):
                action_chunk = self.postprocess(self.policy.predict_action_chunk(batch))
                chunk = action_chunk_to_list(action_chunk)
                return {"action": chunk[0], "action_chunk": chunk}
            action = self.policy.select_action(batch)
            action = self.postprocess(action)
        return {"action": action_to_list(action)}


class SmolVLARequestHandler(BaseHTTPRequestHandler):
    server_version = "SmolVLAServer/0.1"

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _runtime(self) -> SmolVLARuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def _authorized(self, payload: Optional[Dict[str, Any]] = None) -> bool:
        api_key = self.server.api_key  # type: ignore[attr-defined]
        if not api_key:
            return True
        supplied = self.headers.get("X-API-Key", "")
        if not supplied and payload is not None:
            supplied = str(payload.get("api_key", ""))
        return supplied == api_key

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        runtime = self._runtime()
        self._json(
            HTTPStatus.OK,
            {
                "ok": True,
                "model_type": "smolvla",
                "checkpoint": runtime.checkpoint,
                "device": runtime.device,
                "state_key": OBS_STATE,
                "state_dim": runtime.state_dim,
                "image_keys": runtime.image_keys,
                "action_dim": int(np.prod(feature_shape(runtime.policy.config.output_features["action"]))),
            },
        )

    def do_POST(self) -> None:
        if self.path == "/reset":
            self._reset()
            return
        if self.path != "/predict":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "empty_request"})
            return
        if content_length > self.server.max_request_bytes:  # type: ignore[attr-defined]
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large"})
            return

        started = time.time()
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request_body_must_be_json_object")
            if not self._authorized(payload):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            runtime = self._runtime()
            result = runtime.predict(payload)
            self._json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    **result,
                    "action_dim": len(result["action"]),
                    "latency_s": time.time() - started,
                    "request_id": payload.get("request_id"),
                    "session_id": payload.get("session_id", "default"),
                },
            )
        except (ValueError, TypeError, base64.binascii.Error) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:
            LOGGER.error("Prediction request failed: %s\n%s", exc, traceback.format_exc())
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

    def _reset(self) -> None:
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        runtime = self._runtime()
        with runtime.lock:
            runtime.policy.reset()
            runtime.active_session_id = None
        self._json(HTTPStatus.OK, {"ok": True})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s: %(message)s")
    policy, preprocess, postprocess, torch_module = load_policy(args)
    input_features = policy.config.input_features
    if OBS_STATE not in input_features:
        raise ValueError(f"Checkpoint does not define required {OBS_STATE!r} input feature.")
    image_keys = sorted(key for key in input_features if key.startswith(OBS_IMAGES_PREFIX))
    if not image_keys:
        raise ValueError("Checkpoint does not define an observation image input feature.")

    runtime = SmolVLARuntime(
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        torch_module=torch_module,
        checkpoint=args.checkpoint,
        device=args.device,
        state_dim=int(np.prod(feature_shape(input_features[OBS_STATE]))),
        image_keys=image_keys,
        single_image_key=args.single_image_key,
        lock=threading.Lock(),
    )
    runtime.resolve_image_key(runtime.single_image_key)

    server = ThreadingHTTPServer((args.host, args.port), SmolVLARequestHandler)
    server.runtime = runtime  # type: ignore[attr-defined]
    server.api_key = args.api_key  # type: ignore[attr-defined]
    server.max_request_bytes = args.max_request_bytes  # type: ignore[attr-defined]

    LOGGER.info("Serving SmolVLA on http://%s:%d", args.host, args.port)
    LOGGER.info("Expected state: %s[%d]; image features: %s", OBS_STATE, runtime.state_dim, ", ".join(image_keys))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
