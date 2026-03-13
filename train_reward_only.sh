#!/bin/bash
python train.py --env reward_only  --use_cbf_reward_penalty --headless $@
