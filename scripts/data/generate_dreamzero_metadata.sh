#!/bin/bash
set -euo pipefail

cd /meta_eon_cfs/home/Robochallenge/ma/lqc/dreamzero
source .envrc
# conda activate /meta_eon_cfs/home/lqc/miniconda3/envs/dreamzero

DATASET_ROOT=/meta_eon_cfs/home/Robochallenge/dataset/dreamzero_maniparena_final_tabletop_joint
EMBODIMENT_TAG=maniparena_final_tabletop_joint
STATE_KEYS='{"left_ee_cartesian_pos":[0,3],"left_ee_rotation":[3,6],"left_gripper":[6,7],"right_ee_cartesian_pos":[7,10],"right_ee_rotation":[10,13],"right_gripper":[13,14]}'
ACTION_KEYS='{"left_ee_cartesian_pos":[0,3],"left_ee_rotation":[3,6],"left_gripper":[6,7],"right_ee_cartesian_pos":[7,10],"right_ee_rotation":[10,13],"right_gripper":[13,14]}'

for task in \
  pair_up_items \
  pick_fruits_into_basket \
  put_spoon_to_bowl
do
  python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path "$DATASET_ROOT/$task" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --state-keys "$STATE_KEYS" \
    --action-keys "$ACTION_KEYS" \
    --relative-action-keys \
      left_ee_cartesian_pos \
      left_ee_rotation \
      right_ee_cartesian_pos \
      right_ee_rotation \
    --task-key annotation.language.action_text \
    --action-horizon 24 \
    --force
done


# DATASET_ROOT=/meta_eon_cfs/home/Robochallenge/dataset/dreamzero_maniparena_final_mobile_ee
# EMBODIMENT_TAG=maniparena_final_mobile_ee
# STATE_KEYS='{"left_ee_cartesian_pos":[0,3],"left_ee_rotation":[3,6],"left_gripper":[6,7],"right_ee_cartesian_pos":[7,10],"right_ee_rotation":[10,13],"right_gripper":[13,14],"head_actions":[56,58],"height":[58,59],"chassis_velocity":[59,62]}'
# ACTION_KEYS='{"left_ee_cartesian_pos":[0,3],"left_ee_rotation":[3,6],"left_gripper":[6,7],"right_ee_cartesian_pos":[7,10],"right_ee_rotation":[10,13],"right_gripper":[13,14],"head_actions":[56,58],"height":[58,59],"chassis_velocity":[59,62]}'

# for task in \
#   hang_up_picture \
#   organize_shoes \
#   put_bottle_on_woodshelf \
#   put_clothes_in_hamper \
#   take_and_set_tableware
# do
#   python scripts/data/convert_lerobot_to_gear.py \
#     --dataset-path "$DATASET_ROOT/$task" \
#     --embodiment-tag "$EMBODIMENT_TAG" \
#     --state-keys "$STATE_KEYS" \
#     --action-keys "$ACTION_KEYS" \
#     --relative-action-keys \
#       left_ee_cartesian_pos \
#       left_ee_rotation \
#       right_ee_cartesian_pos \
#       right_ee_rotation \
#       head_actions \
#       height \
#     --task-key annotation.language.action_text \
#     --action-horizon 24 \
#     --force
# done
