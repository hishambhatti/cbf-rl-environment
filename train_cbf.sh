#!/bin/bash
python train.py --env cbf --use_cbf_action_filtering --use_cbf_reward_penalty --headless $@
