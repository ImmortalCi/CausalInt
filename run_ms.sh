#seed_list=(1 50 200 3407 5665)
seed_list=(1)

for seed in ${seed_list[@]}
do
python -u main_mindspore_cxy.py \
--train_file data/try_train.csv \
--test_file data/try_test_domain.csv \
--batch_size 1000 \
--learning_rate 1e-3 \
--num_train_epochs 20 \
--gradient_accumulation_steps 1 \
--output_dir ./output/no_pos_weight_epoch=20_lr=1e-3_seed=${seed} \
--seed $seed \
--weight_decay 1e-4 \
--dataloader_num_workers 8
done