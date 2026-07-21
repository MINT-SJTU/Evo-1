#!/bin/bash

policy_name=Evo1
task_name=${1}
task_config=${2:-demo_clean}
ckpt_setting=${3:-step_20000}
seed=${4:-0}
gpu_id=${5:-0}
server_url=${6:-ws://0.0.0.0:9000}
horizon=${7:-37}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning PYTHONUNBUFFERED=1 \
stdbuf -oL python -u script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --server_url ${server_url} \
    --horizon ${horizon}
