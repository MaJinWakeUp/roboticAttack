#!/usr/bin/env python3
"""Serve a LeRobot Pi0.5 checkpoint through the project's JSON/HTTP API.

Run this process on a GPU compute node. ``POST /predict`` accepts a task,
current SO-100/SO-101 joint state, and base64 JPEG images. It returns either
one action or the model's complete action chunk. ``GET /health`` exposes the
checkpoint schema so the robot client can validate it before connecting to the
arm.

Example request body::

    {
      "task": "Stack the red cube on the blue cube",
      "state": [0, 0, 0, 0, 0, 0],
      "images": {
        "up": "<base64 JPEG>",
        "wrist": "<base64 JPEG>"
      },
      "session_id": "so101-1",
      "return_action_chunk": true
    }
"""

import argparse
import logging
import os
import threading
from http import HTTPStatus
from typing import Any

import numpy as np

from experiments.robot.smolvla_server import (
    OBS_IMAGES_PREFIX,
    OBS_STATE,
    SmolVLARequestHandler,
    SmolVLARuntime,
    feature_shape,
)


LOGGER = logging.getLogger("pi05_server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("PI05_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PI05_PORT", "8000")))
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "CHECKPOINT",
            "majinwakeup30/pi05_so101_stack_cube_2_cameras",
        ),
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--single_image_key",
        default=os.environ.get("SINGLE_IMAGE_KEY") or None,
        help="Feature used for image_b64. It is inferred automatically for a one-camera checkpoint.",
    )
    parser.add_argument("--api_key", default=os.environ.get("PI05_API_KEY", ""))
    parser.add_argument("--max_request_bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--log_level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def load_policy(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    """Load Pi0.5 and its checkpoint-specific pre/postprocessors."""
    try:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except ImportError as exc:
        raise ImportError(
            "Pi0.5 serving requires LeRobot's pi and training dependencies. "
            "Install them with `pip install 'lerobot[pi,training]'`."
        ) from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")

    LOGGER.info("Loading Pi0.5 policy from %s", args.checkpoint)
    # Importing PI05Policy above registers the pi05 config discriminator before
    # PreTrainedConfig parses config.json.
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    if getattr(config, "type", None) != "pi05":
        raise ValueError(
            f"Checkpoint {args.checkpoint!r} has policy type {getattr(config, 'type', None)!r}, not 'pi05'."
        )
    config.device = args.device
    config.dtype = args.dtype
    config.gradient_checkpointing = False
    config.compile_model = False

    policy = PI05Policy.from_pretrained(args.checkpoint, config=config).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    return policy, preprocessor, postprocessor, torch


class PI05RequestHandler(SmolVLARequestHandler):
    server_version = "PI05Server/0.1"

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
                "model_type": "pi05",
                "checkpoint": runtime.checkpoint,
                "device": runtime.device,
                "state_key": OBS_STATE,
                "state_dim": runtime.state_dim,
                "image_keys": runtime.image_keys,
                "required_image_keys": runtime.image_keys,
                "action_dim": int(np.prod(feature_shape(runtime.policy.config.output_features["action"]))),
                "chunk_size": int(runtime.policy.config.chunk_size),
                "n_action_steps": int(runtime.policy.config.n_action_steps),
            },
        )


def main() -> None:
    from http.server import ThreadingHTTPServer

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    policy, preprocessor, postprocessor, torch_module = load_policy(args)
    input_features = policy.config.input_features
    if OBS_STATE not in input_features:
        raise ValueError(f"Checkpoint does not define required {OBS_STATE!r} input feature.")
    image_keys = sorted(key for key in input_features if key.startswith(OBS_IMAGES_PREFIX))
    if not image_keys:
        raise ValueError("Checkpoint does not define an observation image input feature.")

    runtime = SmolVLARuntime(
        policy=policy,
        preprocess=preprocessor,
        postprocess=postprocessor,
        torch_module=torch_module,
        checkpoint=args.checkpoint,
        device=args.device,
        state_dim=int(np.prod(feature_shape(input_features[OBS_STATE]))),
        image_keys=image_keys,
        single_image_key=args.single_image_key,
        lock=threading.Lock(),
        model_type="pi05",
    )
    if runtime.single_image_key is None and len(runtime.image_keys) == 1:
        runtime.single_image_key = runtime.image_keys[0]
    if runtime.single_image_key is not None:
        runtime.resolve_image_key(runtime.single_image_key)

    server = ThreadingHTTPServer((args.host, args.port), PI05RequestHandler)
    server.runtime = runtime  # type: ignore[attr-defined]
    server.api_key = args.api_key  # type: ignore[attr-defined]
    server.max_request_bytes = args.max_request_bytes  # type: ignore[attr-defined]
    LOGGER.info("Serving Pi0.5 on http://%s:%d", args.host, args.port)
    LOGGER.info(
        "Expected state: %s[%d]; image features: %s",
        OBS_STATE,
        runtime.state_dim,
        ", ".join(image_keys),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
