# GUI Robot Scripts

Run these from the repo root inside a Palmetto Desktop session:

```bash
cd /home/jin7/projects/VLA/roboticAttack
```

The LIBERO GUI wrappers set `PYTHONPATH` for the local `LIBERO/` checkout, add conda CUDA/cuDNN library paths, and use `vglrun -d egl` when VirtualGL is available. The Bridge wrapper sets the repo `PYTHONPATH` and the same CUDA/cuDNN library paths, but does not use VirtualGL because it shows a Tk camera/control window instead of a MuJoCo viewer.

## Clean OpenVLA GUI

Runs OpenVLA in the visible LIBERO simulator without adversarial patches.

```bash
bash scripts/run_simulation_gui_vla_clean.sh
```

Choose a suite:

```bash
TASK_SUITE=libero_goal bash scripts/run_simulation_gui_vla_clean.sh
TASK_SUITE=libero_object bash scripts/run_simulation_gui_vla_clean.sh
TASK_SUITE=libero_spatial bash scripts/run_simulation_gui_vla_clean.sh
TASK_SUITE=libero_10 bash scripts/run_simulation_gui_vla_clean.sh
```

Choose a task directly instead of using the prompt picker:

```bash
TASK_SUITE=libero_10 TASK_ID=0 bash scripts/run_simulation_gui_vla_clean.sh
```

Override the prompt sent to OpenVLA:

```bash
TASK_SUITE=libero_10 TASK_ID=0 PROMPT="turn on the stove and put the moka pot on it" \
  bash scripts/run_simulation_gui_vla_clean.sh
```

## Attack OpenVLA GUI

Runs the same OpenVLA GUI flow, but applies an adversarial image patch to the camera observation before action prediction.

```bash
bash scripts/run_simulation_gui_vla_attack.sh
```

By default, the script uses the first sorted `patch.pt` under `adversarial_patches/simulation/untargeted`. To choose a patch:

```bash
PATCHROOT=adversarial_patches/simulation/untargeted/UADA-dof1-b55bb4ee-f3df-4410-b7ea-fcfe68ae4132/patch.pt \
  bash scripts/run_simulation_gui_vla_attack.sh
```

Or choose a patch directory:

```bash
PATCH_DIR=adversarial_patches/simulation/untargeted bash scripts/run_simulation_gui_vla_attack.sh
```

Choose suite/task the same way:

```bash
TASK_SUITE=libero_goal TASK_ID=2 PATCHROOT=/path/to/patch.pt \
  bash scripts/run_simulation_gui_vla_attack.sh
```

Patch placement defaults match the existing simulation evaluator:

| Suite | Default patch position |
| --- | --- |
| `libero_10` | `x=5`, `y=160` |
| `libero_object` | `x=30`, `y=150` |
| `libero_goal` | `x=15`, `y=158` |
| `libero_spatial` | `x=120`, `y=160` |

Override patch geometry:

```bash
PATCH_X=5 PATCH_Y=160 PATCH_ANGLE=0 PATCH_SHX=0 PATCH_SHY=0 \
  bash scripts/run_simulation_gui_vla_attack.sh
```

The MuJoCo viewer shows the clean simulator. A separate `Patched OpenVLA Observation` window shows the patched camera image used for OpenVLA action prediction.

Resize that preview, disable it, or turn off the horizontal display correction:

```bash
PATCH_VIEW_SCALE=1 bash scripts/run_simulation_gui_vla_attack.sh
SHOW_PATCH_VIEW=0 bash scripts/run_simulation_gui_vla_attack.sh
PATCH_VIEW_FLIP_X=0 bash scripts/run_simulation_gui_vla_attack.sh
```

When the task succeeds, the attack runner prints a `[success]` line in the terminal and briefly shows the same message in the patched-observation window. To keep that final GUI message longer:

```bash
FINAL_MESSAGE_SECONDS=5 bash scripts/run_simulation_gui_vla_attack.sh
```

## NutNet Attack OpenVLA GUI

Runs the attack GUI with the NutNet DualMask defense from `0502tonylin/NutNet`. The preview window shows four panes: the patched observation with the final mask overlay, the gray-replaced image sent to OpenVLA, the coarse block mask, and the fine pixel mask.

```bash
bash scripts/run_simulation_gui_vla_attack_nutnet.sh
```

By default, NutNet infers fresh masks for every attacked OpenVLA frame. The script uses the downloaded upstream autoencoder weights in `third_party/nutnet/ae_weights_`; with the default `NUTNET_BOX_NUM=32`, it uses `n_13.pth`. Detected pixels are replaced with gray, matching the NutNet defense behavior.

```bash
NUTNET_GRAY_VALUE=128 NUTNET_MASK_OVERLAY_ALPHA=0.45 \
  bash scripts/run_simulation_gui_vla_attack_nutnet.sh
```

The upstream-style thresholds are exposed as environment variables. The defaults follow the YOLOv4 wrapper in NutNet: coarse block reconstruction loss `0.2`, fine pixel reconstruction delta `0.25`, and input size `416`.

```bash
NUTNET_COARSE_THRESHOLD=0.2 NUTNET_FINE_THRESHOLD=0.25 \
  bash scripts/run_simulation_gui_vla_attack_nutnet.sh
```

Use a different NutNet scale or a custom weight file:

```bash
NUTNET_BOX_NUM=16 bash scripts/run_simulation_gui_vla_attack_nutnet.sh
NUTNET_AE_WEIGHTS=/path/to/n_13.pth bash scripts/run_simulation_gui_vla_attack_nutnet.sh
```

If you want the older no-weight heuristic detector, set:

```bash
NUTNET_MODE=heuristic bash scripts/run_simulation_gui_vla_attack_nutnet.sh
```

Use `NUTNET_REFRESH_INTERVAL` to control mask re-detection frequency. `1` means every frame, `20` means every 20 policy frames, and `0` restores the old first-frame mask reuse:

```bash
NUTNET_REFRESH_INTERVAL=20 bash scripts/run_simulation_gui_vla_attack_nutnet.sh
NUTNET_REFRESH_INTERVAL=0 bash scripts/run_simulation_gui_vla_attack_nutnet.sh
```

## Manual Button GUI

Opens the LIBERO simulator plus a button panel for manually moving the arm.

```bash
bash scripts/run_simulation_gui_buttons.sh
```

Choose a suite and select the task interactively:

```bash
TASK_SUITE=libero_object bash scripts/run_simulation_gui_buttons.sh
```

Choose a task directly:

```bash
TASK_SUITE=libero_object TASK_ID=3 bash scripts/run_simulation_gui_buttons.sh
```

Hold movement buttons to keep stepping. Release to stop. The panel includes translation, rotation, gripper, reset, and scale controls.

When the task succeeds, the button panel changes to a `[success]` status message and the same message is printed in the terminal.

## Bridge V2 Physical GUI

Runs OpenVLA against the real-world Bridge V2 / WidowX environment with a Tk control window. This is not a MuJoCo simulator: the GUI shows the live camera image, the predicted action, episode controls, and manual nudge buttons for the physical robot.

This script requires the physical Bridge/WidowX client package that provides `widowx_envs`. If it is not installed in the active environment, install or activate the robot-control environment first. If you already have a local checkout containing `widowx_envs`, point the wrapper at it:

```bash
WIDOWX_ENV_PATH=/path/to/checkout bash scripts/run_bridgev2_gui.sh
```

```bash
bash scripts/run_bridgev2_gui.sh
```

Set the robot service endpoint and task prompt:

```bash
HOST_IP=localhost PORT=5556 TASK="put the carrot on the plate" \
  bash scripts/run_bridgev2_gui.sh
```

Common Bridge options:

```bash
CHECKPOINT=openvla/openvla-7b CUDAID=0 DEVICE=cuda:0 bash scripts/run_bridgev2_gui.sh
CAMERA_TOPIC=/blue/image_raw CONTROL_FREQUENCY=5 MAX_STEPS=60 bash scripts/run_bridgev2_gui.sh
SAVE_DATA=1 SAVE_VIDEO=1 bash scripts/run_bridgev2_gui.sh
```

Tune the manual button step sizes:

```bash
MANUAL_TRANSLATION_STEP=0.005 MANUAL_ROTATION_STEP=0.025 MANUAL_GRIPPER=1.0 \
  bash scripts/run_bridgev2_gui.sh
```

The Reset button calls the WidowX reset flow, which still asks for the start XYZ values in the terminal. Start runs the policy continuously, Step Once executes one OpenVLA action, Pause/Stop prevent further actions, and Mark Success/Mark Failure finish the episode manually. The manual nudge buttons send one Bridge-format action per click: `[dx, dy, dz, droll, dpitch, dyaw, open_gripper]`, where `open_gripper=1` and `close_gripper=0`. Bridge has no automatic `env.check_success()` signal.

## OpenVLA GPU Server + LeRobot Client

Runs OpenVLA on a GPU server and lets a separate client machine send OpenCV camera frames and receive 7-DoF OpenVLA actions. The action returned by the server is `[x, y, z, roll, pitch, yaw, gripper]`.

On the GPU server:

```bash
SERVER_CONDA_ENV=roboticAttack HOST=127.0.0.1 PORT=8000 CUDAID=0 CHECKPOINT=openvla/openvla-7b \
  bash scripts/run_openvla_server.sh
```

`SERVER_CONDA_ENV` defaults to `roboticAttack`, so you can omit it in normal use. Set `SERVER_CONDA_ENV=` only if you have already activated the right environment and want the wrapper to skip conda activation.

From the client, use SSH port forwarding. The client starts `ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 ...` and then talks to `http://127.0.0.1:18000`, so the OpenVLA server never needs to bind to a public interface:

```bash
TASK="put the carrot on the plate" SSH_TUNNEL_HOST=user@gpu-host.example.edu \
  SSH_LOCAL_PORT=18000 SSH_REMOTE_PORT=8000 CAMERA_INDEX=0 STEPS=10 \
  bash scripts/run_lerobot_openvla_client.sh
```

If the GPU host is only reachable through a login/bastion host, add `SSH_TUNNEL_JUMP`:

```bash
TASK="put the carrot on the plate" SSH_TUNNEL_HOST=user@gpu-compute-node \
  SSH_TUNNEL_JUMP=user@login.example.edu SSH_LOCAL_PORT=18000 SSH_REMOTE_PORT=8000 \
  bash scripts/run_lerobot_openvla_client.sh
```

Equivalently, you can create the tunnel yourself in one terminal:

```bash
ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 user@gpu-host.example.edu
```

Then point the client at the local forwarded port:

```bash
TASK="put the carrot on the plate" SERVER_URL=http://127.0.0.1:18000 CAMERA_INDEX=0 STEPS=10 \
  bash scripts/run_lerobot_openvla_client.sh
```

To command a LeRobot SO-100 end-effector robot, pass `EXECUTE=1`. This maps OpenVLA indexes `0,1,2,6` to LeRobot keys `delta_x,delta_y,delta_z,gripper` and converts OpenVLA gripper values from `[0,1]` to SO-100 end-effector `[0,2]`:

```bash
TASK="put the carrot on the plate" SSH_TUNNEL_HOST=user@gpu-host.example.edu \
  ROBOT_TYPE=so100_follower_end_effector ROBOT_PORT=/dev/ttyACM0 \
  URDF_PATH=/path/to/so101_new_calib.urdf EXECUTE=1 \
  bash scripts/run_lerobot_openvla_client.sh
```

For other LeRobot robots, provide an explicit action map before using `EXECUTE=1`:

```bash
ACTION_KEYS=joint_1.pos,joint_2.pos,joint_3.pos ACTION_INDEXES=0,1,2 ACTION_SCALES=1,1,1 \
  ROBOT_TYPE=custom ROBOT_CLASS_PATH=my_robot_pkg:MyRobot ROBOT_CONFIG_CLASS_PATH=my_robot_pkg:MyRobotConfig \
  ROBOT_CONFIG_PATH=/path/to/robot_config.json EXECUTE=1 \
  TASK="open the drawer" SSH_TUNNEL_HOST=user@gpu-host.example.edu bash scripts/run_lerobot_openvla_client.sh
```

Use `MAX_ACTION_ABS` while testing to clip unexpected action values:

```bash
MAX_ACTION_ABS=0.05 EXECUTE=1 bash scripts/run_lerobot_openvla_client.sh
```

## SmolVLA SO-101 GPU Server + Client

The default SmolVLA checkpoint is `Sa74ll/smolvla_so101_pickandplace`, a fine-tuned SO-101 pick-and-place policy. It takes a 6-value state, produces a 6-value joint-target action, and defines `camera1`, `camera2`, and `camera3` inputs. Its published inference example maps the training `up` and `side` views to `camera1` and `camera2`; the third view is optional in the client. The linked dataset metadata labels the robot type `so100_follower`, even though its six joint names and order match SO-101, so verify calibration and use dry runs before enabling motion.

Install LeRobot and the SmolVLA extra once in the GPU server's `roboticAttack` environment. Use the same LeRobot revision as the client when possible:

```bash
conda activate roboticAttack
pip install "lerobot[smolvla]"
```

Start the separate server. On Palmetto, set an API key and bind to `0.0.0.0` so that the login node can forward to the compute node; do not expose an unauthenticated port.

```bash
HOST=0.0.0.0 PORT=8000 CUDAID=0 SMOLVLA_API_KEY='choose-a-secret' \
  CHECKPOINT=Sa74ll/smolvla_so101_pickandplace bash scripts/run_smolvla_server.sh
```

The server exposes `GET /health`, `POST /predict`, and `POST /reset`. A prediction request must contain `task` and `state`; it accepts all three preferred views through `images`, where camera names can be short (`camera1`) or full LeRobot feature names. Add `"return_action_chunk": true` to receive the checkpoint's 50 predicted joint targets in `action_chunk` as well as the first target in `action`:

```json
{
  "task": "put the block in the box",
  "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "images": {
    "camera1": "<base64 JPEG>",
    "camera2": "<base64 JPEG>",
    "camera3": "<base64 JPEG>"
  },
  "session_id": "so101-demo-1",
  "return_action_chunk": true
}
```

The SO-101 client reads the six joint positions from `SO101Follower.get_observation()`, captures the configured OpenCV views, and maps the returned action directly to `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, and `gripper.pos`. By default it requests a 50-action chunk and plays it at 30 Hz, matching the training dataset frame rate without streaming images across SSH every control step. Run a one-step dry run first; this connects and observes but does not command the arm:

```bash
TASK="pick up the cube" SERVER_URL=http://127.0.0.1:18000 SMOLVLA_API_KEY='choose-a-secret' \
  ROBOT_PORT=/dev/ttyACM0 CAMERA1_INDEX=0 CAMERA2_INDEX=1 STEPS=1 \
  bash scripts/run_lerobot_smolvla_so101_client.sh
```

Use the exact task wording and camera placement used for the checkpoint's demonstrations. After checking the returned state and action values, add `EXECUTE=1`; the wrapper defaults `MAX_RELATIVE_TARGET=10` to have LeRobot reject large per-step target changes.

For Palmetto, the client can create the supported login-node tunnel automatically. Replace `node0279` with the compute-node hostname allocated to the server:

```bash
TASK="pick up the cube" ROBOT_PORT=/dev/ttyACM0 CAMERA1_INDEX=0 CAMERA2_INDEX=1 \
  SMOLVLA_API_KEY='choose-a-secret' SSH_TUNNEL_HOST=jin7@slogin.palmetto.clemson.edu \
  SSH_REMOTE_HOST=node0279 SSH_REMOTE_PORT=8000 SSH_LOCAL_PORT=18000 STEPS=1 \
  bash scripts/run_lerobot_smolvla_so101_client.sh
```

## Common Options

Use another GPU:

```bash
CUDAID=1 DEVICE=cuda:0 bash scripts/run_simulation_gui_vla_clean.sh
```

Use a custom checkpoint:

```bash
CHECKPOINT=/path/to/checkpoint TASK_SUITE=libero_goal bash scripts/run_simulation_gui_vla_clean.sh
```

List tasks without running:

```bash
bash scripts/run_simulation_gui_vla_clean.sh --list_tasks
```

Disable VirtualGL wrapper if needed:

```bash
USE_VGL=0 bash scripts/run_simulation_gui_buttons.sh
```

All VLA GUI runners print a dedicated `[success]` line when `env.check_success()` reports that the LIBERO task is complete.

## Task Prompt Reference

Use these task IDs with `TASK_SUITE=... TASK_ID=...`, or leave `TASK_ID` unset and select/search from the terminal prompt.

### `libero_spatial`

| ID | Prompt |
| --- | --- |
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | pick up the black bowl next to the ramekin and place it on the plate |
| 2 | pick up the black bowl from table center and place it on the plate |
| 3 | pick up the black bowl on the cookie box and place it on the plate |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| 5 | pick up the black bowl on the ramekin and place it on the plate |
| 6 | pick up the black bowl next to the cookie box and place it on the plate |
| 7 | pick up the black bowl on the stove and place it on the plate |
| 8 | pick up the black bowl next to the plate and place it on the plate |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate |

### `libero_object`

| ID | Prompt |
| --- | --- |
| 0 | pick up the alphabet soup and place it in the basket |
| 1 | pick up the cream cheese and place it in the basket |
| 2 | pick up the salad dressing and place it in the basket |
| 3 | pick up the bbq sauce and place it in the basket |
| 4 | pick up the ketchup and place it in the basket |
| 5 | pick up the tomato sauce and place it in the basket |
| 6 | pick up the butter and place it in the basket |
| 7 | pick up the milk and place it in the basket |
| 8 | pick up the chocolate pudding and place it in the basket |
| 9 | pick up the orange juice and place it in the basket |

### `libero_goal`

| ID | Prompt |
| --- | --- |
| 0 | open the middle drawer of the cabinet |
| 1 | put the bowl on the stove |
| 2 | put the wine bottle on top of the cabinet |
| 3 | open the top drawer and put the bowl inside |
| 4 | put the bowl on top of the cabinet |
| 5 | push the plate to the front of the stove |
| 6 | put the cream cheese in the bowl |
| 7 | turn on the stove |
| 8 | put the bowl on the plate |
| 9 | put the wine bottle on the rack |

### `libero_10`

| ID | Prompt |
| --- | --- |
| 0 | put both the alphabet soup and the tomato sauce in the basket |
| 1 | put both the cream cheese box and the butter in the basket |
| 2 | turn on the stove and put the moka pot on it |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | pick up the book and place it in the back compartment of the caddy |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | put both moka pots on the stove |
| 9 | put the yellow and white mug in the microwave and close it |
