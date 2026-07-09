#!/usr/bin/env python3
"""
Serve OpenVLA action prediction over a small JSON/HTTP API.

The server is intended to run on a GPU machine. A client sends a JPEG/PNG camera
frame plus a language task, and the server returns OpenVLA's 7-DoF action:
    [x, y, z, roll, pitch, yaw, gripper]
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


LOGGER = logging.getLogger("openvla_server")
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("OPENVLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENVLA_PORT", "8000")))
    parser.add_argument("--pretrained_checkpoint", default=os.environ.get("CHECKPOINT", "openvla/openvla-7b"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--unnorm_key", default=os.environ.get("UNNORM_KEY", "bridge_orig"))
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attn_implementation",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
        default=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"),
    )
    parser.add_argument("--no-attn_fallback", dest="attn_fallback", action="store_false")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--resize_size", type=int, default=0, help="Optional square resize before inference.")
    parser.add_argument("--api_key", default=os.environ.get("OPENVLA_API_KEY", ""))
    parser.add_argument("--max_request_bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--log_level", default=os.environ.get("LOG_LEVEL", "INFO"))
    parser.set_defaults(attn_fallback=True)
    return parser.parse_args()


def register_openvla_auto_classes() -> None:
    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    except ImportError as exc:
        raise ImportError(
            "Could not import OpenVLA Hugging Face classes. Activate the OpenVLA environment "
            "with the repo's pinned dependencies, especially transformers==4.40.1."
        ) from exc

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def torch_dtype(name: str) -> Any:
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def maybe_load_local_norm_stats(model: Any, checkpoint: str) -> None:
    stats_path = Path(checkpoint) / "dataset_statistics.json"
    if not stats_path.is_file():
        return
    with stats_path.open("r") as f:
        model.norm_stats = json.load(f)
    LOGGER.info("Loaded local dataset statistics from %s", stats_path)


def load_model(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    import torch

    register_openvla_auto_classes()
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Choose at most one of --load_in_8bit and --load_in_4bit.")

    dtype = torch_dtype(args.torch_dtype)
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "load_in_8bit": args.load_in_8bit,
        "load_in_4bit": args.load_in_4bit,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    LOGGER.info("Loading processor from %s", args.pretrained_checkpoint)
    processor = AutoProcessor.from_pretrained(args.pretrained_checkpoint, trust_remote_code=True)

    LOGGER.info("Loading OpenVLA model from %s", args.pretrained_checkpoint)
    try:
        model = AutoModelForVision2Seq.from_pretrained(args.pretrained_checkpoint, **model_kwargs)
    except Exception:
        if not args.attn_fallback or args.attn_implementation != "flash_attention_2":
            raise
        LOGGER.warning("flash_attention_2 load failed; retrying with eager attention.", exc_info=True)
        model_kwargs["attn_implementation"] = "eager"
        model = AutoModelForVision2Seq.from_pretrained(args.pretrained_checkpoint, **model_kwargs)

    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(args.device)
    model.eval()
    maybe_load_local_norm_stats(model, args.pretrained_checkpoint)
    return model, processor, torch


def decode_image(image_b64: str) -> Any:
    from PIL import Image

    if "," in image_b64 and image_b64.lstrip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_b64, validate=True)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


def center_crop_image(image: Any, crop_scale: float = 0.9) -> Any:
    from PIL import Image

    width, height = image.size
    crop_ratio = crop_scale**0.5
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BILINEAR)


def build_prompt(checkpoint: str, task: str) -> str:
    task = task.strip().lower()
    if "openvla-v01" in checkpoint:
        return f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task}? ASSISTANT:"
    return f"In: What action should the robot take to {task}?\nOut:"


def action_to_list(action: Any) -> List[float]:
    if hasattr(action, "detach") and hasattr(action, "cpu"):
        action = action.detach().cpu().numpy()
    if isinstance(action, np.ndarray):
        action = action.reshape(-1).tolist()
    if not isinstance(action, (list, tuple)):
        raise TypeError(f"Expected action array/list, got {type(action)}")
    return [float(x) for x in action]


@dataclass
class OpenVLARuntime:
    model: Any
    processor: Any
    torch_module: Any
    checkpoint: str
    device: str
    dtype: Any
    unnorm_key: str
    center_crop: bool
    resize_size: int
    lock: threading.Lock

    def predict(
        self,
        image: Any,
        task: str,
        unnorm_key: Optional[str],
        center_crop: Optional[bool],
    ) -> List[float]:
        from PIL import Image

        do_center_crop = self.center_crop if center_crop is None else bool(center_crop)
        if do_center_crop:
            image = center_crop_image(image)
        if self.resize_size > 0:
            image = image.resize((self.resize_size, self.resize_size), Image.Resampling.BILINEAR)

        prompt = build_prompt(self.checkpoint, task)
        inputs = self.processor(prompt, image).to(self.device, dtype=self.dtype)
        with self.lock, self.torch_module.inference_mode():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=unnorm_key or self.unnorm_key,
                do_sample=False,
            )
        return action_to_list(action)


class OpenVLARequestHandler(BaseHTTPRequestHandler):
    server_version = "OpenVLAServer/0.1"

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _runtime(self) -> OpenVLARuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def _api_key(self) -> str:
        return self.server.api_key  # type: ignore[attr-defined]

    def _authorized(self, payload: Optional[Dict[str, Any]] = None) -> bool:
        api_key = self._api_key()
        if not api_key:
            return True
        supplied = self.headers.get("X-API-Key", "")
        if payload is not None and not supplied:
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
        norm_keys = sorted(getattr(runtime.model, "norm_stats", {}).keys())
        self._json(
            HTTPStatus.OK,
            {
                "ok": True,
                "checkpoint": runtime.checkpoint,
                "device": runtime.device,
                "unnorm_key": runtime.unnorm_key,
                "norm_keys": norm_keys,
            },
        )

    def do_POST(self) -> None:
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
            if not self._authorized(payload):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return

            task = str(payload.get("task", "")).strip()
            image_b64 = payload.get("image_b64") or payload.get("image")
            if not task:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "missing_task"})
                return
            if not image_b64:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "missing_image_b64"})
                return

            image = decode_image(str(image_b64))
            runtime = self._runtime()
            action = runtime.predict(
                image=image,
                task=task,
                unnorm_key=payload.get("unnorm_key"),
                center_crop=payload.get("center_crop"),
            )
            latency_s = time.time() - started
            self._json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "action": action,
                    "action_dim": len(action),
                    "latency_s": latency_s,
                    "unnorm_key": payload.get("unnorm_key") or runtime.unnorm_key,
                    "request_id": payload.get("request_id"),
                },
            )
        except Exception as exc:
            LOGGER.error("Prediction request failed: %s\n%s", exc, traceback.format_exc())
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(asctime)s] %(levelname)s: %(message)s")

    model, processor, torch_module = load_model(args)
    runtime = OpenVLARuntime(
        model=model,
        processor=processor,
        torch_module=torch_module,
        checkpoint=args.pretrained_checkpoint,
        device=args.device,
        dtype=torch_dtype(args.torch_dtype),
        unnorm_key=args.unnorm_key,
        center_crop=args.center_crop,
        resize_size=args.resize_size,
        lock=threading.Lock(),
    )

    server = ThreadingHTTPServer((args.host, args.port), OpenVLARequestHandler)
    server.runtime = runtime  # type: ignore[attr-defined]
    server.api_key = args.api_key  # type: ignore[attr-defined]
    server.max_request_bytes = args.max_request_bytes  # type: ignore[attr-defined]

    LOGGER.info("Serving OpenVLA on http://%s:%d", args.host, args.port)
    LOGGER.info("Health check: curl http://%s:%d/health", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
