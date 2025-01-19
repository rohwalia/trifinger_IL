# python3 imitate_episodes.py \
# --task_name lift \
# --ckpt_dir "/home/local_rohan/act/checkpoints/lift" \
# --policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
# --num_epochs 2000  --lr 1e-5 \
# --seed 0

python3 imitate_episodes.py \
--task_name lift \
--ckpt_dir "/home/local_rohan/act/checkpoints_mlp/lift" \
--policy_class CNNMLP --batch_size 8 \
--num_epochs 2000  --lr 1e-5 \
--seed 0