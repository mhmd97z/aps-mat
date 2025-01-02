#!/bin/sh
env="Aps"
algo="mat"
exp="sinrdb1_powerdb1_sumcost0_localpowersum0_scoef20_pcoef1_1ue_10ap_sinrn10"
seed=1
n_rollout_threads=32
echo "env is ${env}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_aps.py --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 16 \
 --n_rollout_threads ${n_rollout_threads} --num_mini_batch 1 --episode_length 10 \ 
 --num_env_steps 10000000 --lr 5e-4 --ppo_epoch 15 --clip_param 0.05 --save_interval 100000 \
 --use_value_active_masks --use_eval
