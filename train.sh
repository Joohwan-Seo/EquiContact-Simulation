python preprocess_data.py --config_file "config/train/ACT_stack_08.yaml" 

python act/train_act.py --task_name "sim_stack_v0" \
--policy_class "ACT" --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
--num_epochs 8000 --lr 1e-05  --seed 0 --temporal_agg --config_file "config/train/ACT_stack_08.yaml"

