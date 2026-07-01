# A Real Training Sample (roboticAttack / OpenVLA patch attack)

This walks one concrete example all the way from the on-disk dataset to the exact
tensors fed into OpenVLA during patch optimization. Values are grounded in:

- Dataset schema: `roboticAttack/dataset/libero_object_no_noops/1.0.0/features.json`
- Transform: `roboticAttack/VLAAttacker/white_patch/RLDSBatchTransform.py`
- Prompt format: `roboticAttack/prismatic/models/backbones/llm/prompting/base_prompter.py`
- Action tokenizer: `roboticAttack/prismatic/vla/action_tokenizer.py`
- Task instruction: `roboticAttack/LIBERO/.../libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl`

Dataset used here: **libero_object_no_noops** (a simulation suite). Model the patch
is attacking: **openvla/openvla-7b-finetuned-libero-object**.

---

## STAGE 0 — One raw RLDS step (as stored in the .tfrecord)

A dataset is a set of *episodes*; each episode is a sequence of *steps*. One step
(one timestep of one demonstration) looks like this:

```jsonc
{
  "language_instruction": "pick up the tomato sauce and place it in the basket",

  "observation": {
    "image":       "<uint8 array, shape [256, 256, 3]>",   // main 3rd-person camera (RGB)
    "wrist_image": "<uint8 array, shape [256, 256, 3]>",   // wrist camera (unused by this attack)
    "state":       [ 0.034, -0.118,  0.197,  3.071, -0.044,  0.015,  0.021, -0.021 ], // 6D EEF pose + 2D gripper
    "joint_state": [-0.021,  0.118, -0.004, -2.451,  0.013,  2.557,  0.842 ]          // 7 joint angles
  },

  // 7-DoF action = the LABEL the model must predict (delta end-effector + gripper)
  //        [  dx ,   dy ,   dz , droll, dpitch, dyaw, gripper ]
  "action": [ 0.12, -0.05,  0.34,  0.00,   0.00,  0.00,  1.00 ],

  "reward": 0.0, "discount": 1.0,
  "is_first": true, "is_last": false, "is_terminal": false
}
```

> Only `observation.image`, `language_instruction`, and `action` are used by the
> attack. `wrist_image`, `state`, `joint_state`, reward, etc. are ignored.

The "**input**" to the model = the main camera image + the language instruction.
The "**target / label**" = the 7-DoF action vector (what a successful demo did).

---

## STAGE 1 — Build the text prompt (RLDSBatchTransform)

The instruction is lowercased and wrapped into OpenVLA's fixed chat template
(`In: ... \nOut: ...`). The 7 action floats are discretized into 7 action tokens
and become the model's expected output:

```
<s>In: What action should the robot take to pick up the tomato sauce and place it in the basket?
Out: ⟨a0⟩⟨a1⟩⟨a2⟩⟨a3⟩⟨a4⟩⟨a5⟩⟨a6⟩</s>
```

- `In: ... Out: ` = the human turn (the **prompt** / context).
- `⟨a0⟩…⟨a6⟩</s>` = the gpt turn (the **answer** the model is trained/attacked to produce).

---

## STAGE 2 — Action → tokens (action_tokenizer.py)

Each action dimension is clipped to [-1, 1], split into **256 uniform bins**, and
the bin index is mapped to the **last 256 token IDs** of the Llama vocab
(`token_id = 32000 - bin_index`). `action_token_begin_idx = 31743`, so every action
token is in the range **[31744, 31999]**.

For `action = [0.12, -0.05, 0.34, 0.00, 0.00, 0.00, 1.00]`:

| dim | meaning  | value | bin index | token id |
|-----|----------|-------|-----------|----------|
| a0  | dx       | 0.12  | 143       | **31857** |
| a1  | dy       | -0.05 | 122       | **31878** |
| a2  | dz       | 0.34  | 171       | **31829** |
| a3  | droll    | 0.00  | 128       | **31872** |
| a4  | dpitch   | 0.00  | 128       | **31872** |
| a5  | dyaw     | 0.00  | 128       | **31872** |
| a6  | gripper  | 1.00  | 256       | **31744** |

(Note: token `31872` == action value 0, and `31744` == the max/close-gripper value —
the same magic numbers hard-coded in the attack loss functions.)

Round-trip decode of these tokens = `[0.118, -0.047, 0.337, 0.0, 0.0, 0.0, 0.996]`
(small error is just the bin quantization).

---

## STAGE 3 — The actual training tensors (one sample)

After tokenization + the collator, this is one element of the batch
(`bs=8` such elements are stacked):

```python
sample = {
  # --- INPUT IMAGE ---
  # PIL image -> resized to 224x224 -> normalized TWICE (DINOv2 + SigLIP stats)
  # -> concatenated on channel dim => 6 channels. bfloat16.
  "pixel_values":   Tensor(shape=[6, 224, 224], dtype=torch.bfloat16),

  # --- INPUT TEXT (prompt + answer tokens, BOS prepended) ---
  "input_ids":      tensor([    1,  512,   29901, ... ,   29991,   13,  4905, 29901,
                              31857, 31878, 31829, 31872, 31872, 31872, 31744,     2]),
  #                  ^^^=<s>(BOS)        ... prompt words ...     ^^^^ 7 action tokens ^^ </s>(EOS=2)

  "attention_mask": tensor([1, 1, 1, ... , 1, 1, 1, 1, 1, 1, 1, 1, 1]),   # all real tokens = 1

  # --- LABELS ---  loss is masked everywhere EXCEPT the 7 action tokens + EOS
  "labels":         tensor([ -100, -100, -100, ... , -100, -100,
                            31857, 31878, 31829, 31872, 31872, 31872, 31744,    2]),
  #                  ^^^^^^^ prompt positions = IGNORE_INDEX (-100) ^^^^^^^   ^^^ supervised ^^^
}
```

Key line that masks the prompt (RLDSBatchTransform.py:45):
`labels[: -(len(action) + 1)] = -100`  → only the final `7 actions + 1 EOS` carry loss.

---

## STAGE 4 — What the ATTACK does to this sample

The image and instruction are **never changed**; the patch and the *labels* are what
the attack manipulates.

1. **Patch is composited into the image** before the forward pass
   (`apply_random_patch_batch`): the learnable `[3, H, W]` patch is pasted at a
   **random location** with **random rotation (±30°) + shear (±0.2)** (EOT), then the
   result is normalized into the 6-channel `pixel_values` above.

2. **The label is overwritten depending on the attack** (here, targeting dims
   `--maskidx 0` = dx only):

   - **TMA (targeted)** — force a chosen action, e.g. `--targetAction 0`:
     ```
     labels' action tokens = [31872(=0), -100, -100, -100, -100, -100, -100]
     # minimize CE so the model outputs dx = 0 regardless of the scene
     ```
   - **UADA (untargeted)** — keep only dim 0, push it to the OPPOSITE extreme of GT:
     ```
     GT dx token = 31857 (>31872 region)  -> loss target = lowest bin  (drive dx far negative)
     ```
   - **UPA (position-aware)** — flip the xyz translation direction via a
     cosine-similarity + distance loss on dims [dx, dy, dz].

3. **Only the patch pixels get gradients.** `loss.backward()` → `optimizer.step()`
   updates the patch → `patch.clamp(0,1)`. The OpenVLA weights stay frozen.

So for THIS sample, a successful TMA(dx→0) patch makes OpenVLA emit
`dx ≈ 0` (token 31872) even though the correct demonstrated action was `dx = 0.12`
(token 31857) — i.e. the robot fails to move toward the tomato sauce.
