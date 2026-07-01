# GUI Simulation Scripts

Run these from the repo root inside a Palmetto Desktop session:

```bash
cd /home/jin7/projects/VLA/roboticAttack
```

All GUI wrappers set `PYTHONPATH` for the local `LIBERO/` checkout, add conda CUDA/cuDNN library paths, and use `vglrun -d egl` when VirtualGL is available.

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
