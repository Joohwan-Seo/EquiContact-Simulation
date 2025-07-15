python act/eval_act.py --task_name "sim_stack_v0" \
--policy_class "ACT" --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
--num_epochs 6000 --lr 1e-05  --seed 0 --config_file "config/train/ACT_stack_09.yaml" --temporal_agg
