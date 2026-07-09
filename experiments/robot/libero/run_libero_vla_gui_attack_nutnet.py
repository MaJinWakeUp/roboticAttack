"""
Run one LIBERO task in a visible MuJoCo viewer with OpenVLA control under an
adversarial image patch and a NutNet-style per-frame defense.

The simulator itself is clean. The patch is applied to the camera image that is
sent to OpenVLA before action prediction. NutNet's coarse block mask and fine
pixel mask are inferred from each attacked frame by default. Detected pixels are
replaced with gray before action prediction.
"""

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
from experiments.robot.libero.run_libero_vla_gui_attack import (
    MAX_STEPS,
    apply_patch,
    check_unnorm_key,
    choose_task_id,
    list_tasks,
    load_patch,
    make_cfg,
    make_gui_env,
    render,
    reset_env,
    resolve_checkpoint,
    resolve_patch_position,
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
    parser.add_argument("--final_message_seconds", type=float, default=2.0)
    parser.add_argument("--nutnet_box_num", type=int, default=32)
    parser.add_argument("--nutnet_input_size", type=int, default=416)
    parser.add_argument("--nutnet_mode", choices=("autoencoder", "heuristic"), default="autoencoder")
    parser.add_argument("--nutnet_ae_weights", default=None)
    parser.add_argument("--nutnet_device", default="auto")
    parser.add_argument("--nutnet_coarse_threshold", type=float, default=0.2)
    parser.add_argument("--nutnet_fine_threshold", type=float, default=0.25)
    parser.add_argument("--nutnet_threshold_scale", type=float, default=4.0)
    parser.add_argument("--nutnet_max_mask_fraction", type=float, default=1.0)
    parser.add_argument("--nutnet_blur_kernel", type=int, default=5)
    parser.add_argument("--nutnet_refresh_interval", type=int, default=1)
    parser.add_argument("--nutnet_gray_value", type=int, default=128)
    parser.add_argument("--nutnet_mask_overlay_alpha", type=float, default=0.45)
    return parser.parse_args()


class AutoEncoder8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=8, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 3, kernel_size=8, stride=2, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoEncoder16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 3, kernel_size=2, stride=2, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoEncoder32(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 3, kernel_size=2, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class NutNetVideoDefense:
    def __init__(
        self,
        box_num=32,
        input_size=416,
        mode="autoencoder",
        ae_weights=None,
        device="cpu",
        coarse_threshold=0.2,
        fine_threshold=0.25,
        threshold_scale=4.0,
        max_mask_fraction=1.0,
        blur_kernel=5,
        refresh_interval=1,
        gray_value=128,
    ):
        if box_num <= 0:
            raise ValueError("nutnet_box_num must be positive")
        if input_size <= 0 or input_size % box_num != 0:
            raise ValueError("nutnet_input_size must be positive and divisible by nutnet_box_num")
        if blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ValueError("nutnet_blur_kernel must be an odd positive integer")
        self.box_num = box_num
        self.input_size = input_size
        self.box_length = input_size // box_num
        self.mode = mode
        self.device = torch.device(device)
        self.coarse_threshold = coarse_threshold
        self.fine_threshold = fine_threshold
        self.threshold_scale = threshold_scale
        self.max_mask_fraction = max(0.0, min(1.0, max_mask_fraction))
        self.blur_kernel = blur_kernel
        self.refresh_interval = max(0, refresh_interval)
        self.gray_value = max(0, min(255, gray_value)) / 255.0
        self.ae = None
        self.cached_masks = None
        self.frame_index = 0
        if self.mode == "autoencoder":
            self.ae = self._load_autoencoder(ae_weights)

    def _default_weight_path(self):
        return REPO_ROOT / "third_party" / "nutnet" / "ae_weights_" / f"n_{self.box_length}.pth"

    def _make_autoencoder(self):
        if self.box_num == 8:
            return AutoEncoder8()
        if self.box_num == 16:
            return AutoEncoder16()
        if self.box_num == 32:
            return AutoEncoder32()
        raise ValueError("NutNet autoencoder mode supports box_num 8, 16, or 32")

    def _load_autoencoder(self, ae_weights):
        weights_path = Path(ae_weights) if ae_weights else self._default_weight_path()
        if not weights_path.is_absolute():
            weights_path = REPO_ROOT / weights_path
        if not weights_path.is_file():
            print(
                f"[warn] NutNet AE weights not found at {weights_path}; falling back to heuristic masks.",
                flush=True,
            )
            self.mode = "heuristic"
            return None

        model = self._make_autoencoder().to(self.device)
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"[nutnet] loaded AE weights from {weights_path}", flush=True)
        return model

    def _to_tensor(self, image):
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-1] == 3 and tensor.shape[1] != 3:
                tensor = tensor.permute(0, 3, 1, 2)
            if tensor.dtype.is_floating_point:
                if tensor.max() > 1.5:
                    tensor = tensor / 255.0
            else:
                tensor = tensor.float() / 255.0
            return tensor.clamp(0.0, 1.0)

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected an HWC RGB image, got shape {array.shape}")
        tensor = torch.from_numpy(array).float().permute(2, 0, 1).unsqueeze(0)
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    def _to_numpy(self, tensor):
        array = tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
        return array

    def _mask_blocks_to_pixels(self, block_mask):
        block_mask = block_mask.view(1, self.box_num, self.box_num, 1, 1)
        block_mask = block_mask.expand(1, self.box_num, self.box_num, self.box_length, self.box_length)
        return block_mask.permute(0, 1, 3, 2, 4).contiguous().view(1, 1, self.input_size, self.input_size).bool()

    def _blocks_to_image(self, pixel_blocks):
        pixel_blocks = pixel_blocks.view(1, self.box_num, self.box_num, self.box_length, self.box_length)
        return pixel_blocks.permute(0, 1, 3, 2, 4).contiguous().view(1, 1, self.input_size, self.input_size)

    def _mask_pixels_from_blocks(self, pixel_blocks):
        return self._blocks_to_image(pixel_blocks).bool()

    def _resize_mask(self, mask, height, width):
        mask = F.interpolate(mask.float(), size=(height, width), mode="nearest")
        return mask.bool()

    def _cap_mask(self, mask, scores=None):
        if self.max_mask_fraction >= 1.0:
            return mask
        max_mask_count = max(1, int(round(self.max_mask_fraction * mask.numel())))
        if mask.sum().item() <= max_mask_count:
            return mask

        flat_mask = mask.flatten()
        if scores is None:
            scores = flat_mask.float()
        flat_scores = scores.flatten()
        candidates = torch.where(flat_mask, flat_scores, torch.full_like(flat_scores, -1.0))
        topk = torch.topk(candidates, max_mask_count).indices
        capped = torch.zeros_like(flat_mask, dtype=torch.bool)
        capped[topk] = True
        return capped.view_as(mask)

    def _split_blocks(self, image_tensor):
        image_tensor = image_tensor * 2.0 - 1.0
        blocks = image_tensor.unfold(2, self.box_length, self.box_length).unfold(3, self.box_length, self.box_length)
        blocks = blocks.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, self.box_length, self.box_length, 3)
        return blocks.permute(0, 3, 1, 2).contiguous()

    def _infer_autoencoder_masks(self, image_tensor):
        if self.ae is None:
            return self._infer_heuristic_masks(image_tensor)

        height, width = image_tensor.shape[-2:]
        model_input = F.interpolate(
            image_tensor.to(self.device),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        blocks = self._split_blocks(model_input)
        with torch.no_grad():
            output = self.ae(blocks)
        if output.shape != blocks.shape:
            raise RuntimeError(f"NutNet AE output shape {tuple(output.shape)} did not match {tuple(blocks.shape)}")

        loss = F.mse_loss(output, blocks, reduction="none").mean(dim=(1, 2, 3))
        coarse_blocks = (loss > self.coarse_threshold).view(self.box_num, self.box_num)
        coarse_mask_model = self._mask_blocks_to_pixels(coarse_blocks)

        delta = torch.abs(output - blocks)
        fine_blocks = delta.sum(dim=1) > self.fine_threshold
        fine_mask_model = self._mask_pixels_from_blocks(fine_blocks)

        final_mask_model = coarse_mask_model & fine_mask_model
        if self.max_mask_fraction < 1.0:
            fine_scores = self._blocks_to_image(delta.sum(dim=1))
            final_mask_model = self._cap_mask(final_mask_model, fine_scores)

        return {
            "coarse": self._resize_mask(coarse_mask_model, height, width).cpu(),
            "fine": self._resize_mask(fine_mask_model, height, width).cpu(),
            "final": self._resize_mask(final_mask_model, height, width).cpu(),
        }

    def _infer_heuristic_block_mask(self, image_tensor):
        blur = F.avg_pool2d(image_tensor, kernel_size=self.blur_kernel, stride=1, padding=self.blur_kernel // 2)
        residual = (image_tensor - blur).abs().mean(dim=1, keepdim=True)
        block_scores = F.adaptive_avg_pool2d(residual, (self.box_num, self.box_num)).squeeze(0).squeeze(0)

        median = block_scores.median()
        mad = (block_scores - median).abs().median().clamp_min(1e-6)
        threshold = median + self.threshold_scale * mad
        mask = block_scores > threshold

        if not bool(mask.any()):
            mask = block_scores >= block_scores.max()

        max_mask_count = max(1, int(round(self.max_mask_fraction * mask.numel())))
        if mask.sum().item() > max_mask_count:
            topk = torch.topk(block_scores.flatten(), max_mask_count).indices
            flat_mask = torch.zeros_like(block_scores.flatten(), dtype=torch.bool)
            flat_mask[topk] = True
            mask = flat_mask.view_as(block_scores)

        return mask

    def _infer_heuristic_masks(self, image_tensor):
        height, width = image_tensor.shape[-2:]
        block_mask = self._infer_heuristic_block_mask(image_tensor)
        coarse_mask = self._resize_mask(block_mask.float().view(1, 1, self.box_num, self.box_num), height, width)

        blur = F.avg_pool2d(image_tensor, kernel_size=self.blur_kernel, stride=1, padding=self.blur_kernel // 2)
        residual = (image_tensor - blur).abs().mean(dim=1, keepdim=True)
        median = residual.median()
        mad = (residual - median).abs().median().clamp_min(1e-6)
        fine_mask = residual > (median + self.threshold_scale * mad)
        final_mask = coarse_mask & fine_mask
        if not bool(final_mask.any()):
            final_mask = coarse_mask

        return {"coarse": coarse_mask, "fine": fine_mask, "final": final_mask}

    def _infer_masks(self, image_tensor):
        if self.mode == "autoencoder":
            return self._infer_autoencoder_masks(image_tensor)
        return self._infer_heuristic_masks(image_tensor)

    def __call__(self, image):
        image_tensor = self._to_tensor(image)
        if self.cached_masks is None or (
            self.refresh_interval > 0 and self.frame_index % self.refresh_interval == 0 and self.frame_index > 0
        ):
            self.cached_masks = self._infer_masks(image_tensor)
            final_coverage = self.cached_masks["final"].float().mean().item()
            coarse_coverage = self.cached_masks["coarse"].float().mean().item()
            fine_coverage = self.cached_masks["fine"].float().mean().item()
            print(
                f"[nutnet] inferred {self.mode} masks on frame {self.frame_index} "
                f"(coarse={coarse_coverage:.3f}, fine={fine_coverage:.3f}, final={final_coverage:.3f})",
                flush=True,
            )

        pixel_mask = self.cached_masks["final"]
        gray = torch.full_like(image_tensor, self.gray_value)
        defended = torch.where(pixel_mask, gray, image_tensor)
        masks_np = {
            name: mask.squeeze(0).squeeze(0).cpu().numpy().astype(bool)
            for name, mask in self.cached_masks.items()
        }
        mask_fraction = pixel_mask.float().mean().item()

        self.frame_index += 1
        return self._to_numpy(defended), masks_np, mask_fraction


class NutNetViewer:
    def __init__(self, enabled, scale, flip_x, overlay_alpha):
        self.root = None
        self.patched_label = None
        self.coarse_label = None
        self.fine_label = None
        self.defended_label = None
        self.status_label = None
        self.patched_photo = None
        self.coarse_photo = None
        self.fine_photo = None
        self.defended_photo = None
        self.scale = scale
        self.flip_x = flip_x
        self.overlay_alpha = max(0.0, min(1.0, overlay_alpha))
        if not enabled:
            return

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            print(f"[warn] could not open NutNet view GUI: {exc}")
            return

        self.root.title("NutNet Patch / Mask / Defense Preview")
        grid = tk.Frame(self.root)
        grid.pack(padx=8, pady=(8, 4))

        tk.Label(grid, text="Patched + final overlay").grid(row=0, column=0, padx=4, pady=(0, 4))
        tk.Label(grid, text="Gray-replaced defense").grid(row=0, column=1, padx=4, pady=(0, 4))
        tk.Label(grid, text="Coarse block mask").grid(row=2, column=0, padx=4, pady=(8, 4))
        tk.Label(grid, text="Fine pixel mask").grid(row=2, column=1, padx=4, pady=(8, 4))

        self.patched_label = tk.Label(grid)
        self.patched_label.grid(row=1, column=0, padx=4)
        self.defended_label = tk.Label(grid)
        self.defended_label.grid(row=1, column=1, padx=4)
        self.coarse_label = tk.Label(grid)
        self.coarse_label.grid(row=3, column=0, padx=4)
        self.fine_label = tk.Label(grid)
        self.fine_label.grid(row=3, column=1, padx=4)

        self.status_label = tk.Label(self.root, text="Waiting for first defended observation")
        self.status_label.pack(padx=8, pady=(0, 8))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _display_photo(self, img):
        display_img = np.fliplr(img) if self.flip_x else img
        image = Image.fromarray(display_img).convert("RGB")
        if self.scale != 1.0:
            width = max(1, int(image.width * self.scale))
            height = max(1, int(image.height * self.scale))
            image = image.resize((width, height), Image.Resampling.NEAREST)
        return ImageTk.PhotoImage(image)

    def _make_mask_image(self, mask):
        mask_img = mask.astype(np.uint8) * 255
        return np.stack([mask_img, mask_img, mask_img], axis=-1)

    def _make_overlay_image(self, img, mask):
        overlay = img.copy()
        color = np.array([255, 0, 0], dtype=np.float32)
        mask_pixels = mask.astype(bool)
        if mask_pixels.any():
            overlay[mask_pixels] = (
                (1.0 - self.overlay_alpha) * overlay[mask_pixels].astype(np.float32)
                + self.overlay_alpha * color
            ).round().astype(np.uint8)
        return overlay

    def update(self, attacked_img, masks, defended_img, step_idx, mask_fraction):
        if self.root is None:
            return
        overlay_img = self._make_overlay_image(attacked_img, masks["final"])
        coarse_img = self._make_mask_image(masks["coarse"])
        fine_img = self._make_mask_image(masks["fine"])

        self.patched_photo = self._display_photo(overlay_img)
        self.defended_photo = self._display_photo(defended_img)
        self.coarse_photo = self._display_photo(coarse_img)
        self.fine_photo = self._display_photo(fine_img)
        self.patched_label.configure(image=self.patched_photo)
        self.defended_label.configure(image=self.defended_photo)
        self.coarse_label.configure(image=self.coarse_photo)
        self.fine_label.configure(image=self.fine_photo)
        self.status_label.configure(
            text=f"NutNet DualMask gray replacement, step {step_idx}, final mask pixels={mask_fraction:.3f}"
        )
        self.root.update_idletasks()
        self.root.update()

    def set_status(self, text):
        if self.root is None:
            return
        self.status_label.configure(text=text)
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        if self.root is None:
            return
        root = self.root
        self.root = None
        root.destroy()


def run_policy(args, cfg, model, processor, env, task_suite, task_id, prompt, patch, position, viewer):
    resize_size = get_image_resize_size(cfg)
    patch_transform = RandomPatchTransform("cpu", False)
    nutnet_device = args.nutnet_device
    if nutnet_device == "auto":
        nutnet_device = args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    nutnet = NutNetVideoDefense(
        box_num=args.nutnet_box_num,
        input_size=args.nutnet_input_size,
        mode=args.nutnet_mode,
        ae_weights=args.nutnet_ae_weights,
        device=nutnet_device,
        coarse_threshold=args.nutnet_coarse_threshold,
        fine_threshold=args.nutnet_fine_threshold,
        threshold_scale=args.nutnet_threshold_scale,
        max_mask_fraction=args.nutnet_max_mask_fraction,
        blur_kernel=args.nutnet_blur_kernel,
        refresh_interval=args.nutnet_refresh_interval,
        gray_value=args.nutnet_gray_value,
    )
    max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS.get(args.task_suite_name, 520)
    obs = reset_env(env, task_suite, task_id, args.init_state_id)

    for t in range(max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            render(env)
            continue

        attacked_img = get_libero_image(obs, resize_size)
        attacked_img = apply_patch(attacked_img, patch_transform, patch, args, position)
        defended_img, masks, mask_fraction = nutnet(attacked_img)
        viewer.update(attacked_img, masks, defended_img, t - args.num_steps_wait + 1, mask_fraction)

        observation = {
            "full_image": defended_img,
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
        policy_step = t - args.num_steps_wait + 1
        print(f"step={policy_step} reward={reward:.3f} done={done} success={success}")
        if success or done:
            if success:
                message = f"[success] Task succeeded at policy step {policy_step} (reward={reward:.3f})."
            else:
                message = f"[done] Episode ended at policy step {policy_step} before success (reward={reward:.3f})."
            print(message, flush=True)
            viewer.set_status(message)
            if viewer.root is not None and args.final_message_seconds > 0:
                time.sleep(args.final_message_seconds)
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
    print(
        f"NutNet: mode={args.nutnet_mode} box_num={args.nutnet_box_num} input_size={args.nutnet_input_size} "
        f"coarse_threshold={args.nutnet_coarse_threshold} fine_threshold={args.nutnet_fine_threshold} "
        f"threshold_scale={args.nutnet_threshold_scale} max_mask_fraction={args.nutnet_max_mask_fraction} "
        f"blur_kernel={args.nutnet_blur_kernel} refresh_interval={args.nutnet_refresh_interval} "
        f"gray_value={args.nutnet_gray_value} ae_weights={args.nutnet_ae_weights or 'auto'} "
        f"mask_overlay_alpha={args.nutnet_mask_overlay_alpha}"
    )

    env = None
    viewer = NutNetViewer(
        args.show_patch_view,
        args.patch_view_scale,
        args.patch_view_flip_x,
        args.nutnet_mask_overlay_alpha,
    )
    try:
        model = get_model(cfg, DEVICE=args.device)
        model.eval()
        check_unnorm_key(cfg, model)
        processor = get_processor(cfg)
        env = make_gui_env(args, task)
        run_policy(args, cfg, model, processor, env, task_suite, task_id, prompt, patch, position, viewer)
    finally:
        viewer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
