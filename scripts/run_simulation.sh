#!/bin/bash
python evaluation_tool/eval_queue_single_four_spec.py \
    --exp_path adversarial_patches/simulation/untargeted \
    --cudaid 0 \
    --trials 50 \
    --max_concurrent_tasks 1 \
    --task libero_10 \     
