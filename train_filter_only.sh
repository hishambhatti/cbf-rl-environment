#!/bin/bash
python train.py --env filter_only --use_cbf_action_filtering --headless $@
