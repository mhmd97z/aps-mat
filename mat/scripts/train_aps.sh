#!/bin/sh
env="Aps"
algo="mat"
exp="scen2_base"
echo "env is ${env}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_aps.py --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed 1 --n_training_threads 16 --gamma 0.01 --use_wandb False \
 --n_rollout_threads 16 --num_mini_batch 1 --episode_length 10 --use_valuenorm \
 --ppo_epoch 15 --clip_param 0.2 --max_grad_norm 0.5 \
 --lr 0.0005 --critic_lr 0.0005 --entr_lr 0.001 \
 --num_env_steps 1000000
