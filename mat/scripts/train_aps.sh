#!/bin/sh
env="Aps"
algo="mat"
exp="partial_olp/ap20_ue6_sinr0/ped_10step_50ms/mat"
python train/train_aps.py --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed 1 --n_training_threads 16 --gamma 0.01 --use_wandb False \
 --n_rollout_threads 16 --num_mini_batch 1 --episode_length 100 --use_valuenorm \
 --ppo_epoch 5 --clip_param 0.2 --max_grad_norm 0.5 \
 --lr 0.0001 --critic_lr 0.0001  --use_linear_lr_decay \
 --num_env_steps 300000 --entropy_coef 0.0005 --log_interval 1
