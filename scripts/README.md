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

## Manual Button GUI

Opens the LIBERO simulator plus a button panel for manually moving the arm.

```bash
bash scripts/run_simulation_gui_buttons.sh
```

Choose suite/task:

```bash
TASK_SUITE=libero_object TASK_ID=3 bash scripts/run_simulation_gui_buttons.sh
```

Hold movement buttons to keep stepping. Release to stop. The panel includes translation, rotation, gripper, reset, and scale controls.

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
