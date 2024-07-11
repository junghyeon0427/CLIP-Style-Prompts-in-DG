python3 PIN_aug.py \
--dataset cityscapes \
--data_root /workspace/cityscapes \
--total_it 100 \
--resize_feat \
--target_domain_desc "Driving at night" \
--save_dir result_night \
--batch_size 40

python3 PIN_aug.py \
--dataset cityscapes \
--data_root /workspace/cityscapes \
--total_it 100 \
--resize_feat \
--target_domain_desc "Driving in snow" \
--save_dir result_snow \
--batch_size 40

python3 PIN_aug.py \
--dataset cityscapes \
--data_root /workspace/cityscapes \
--total_it 100 \
--resize_feat \
--target_domain_desc "Driving under rain" \
--save_dir result_rain \
--batch_size 40

python3 PIN_aug.py \
--dataset cityscapes \
--data_root /workspace/cityscapes \
--total_it 100 \
--resize_feat \
--target_domain_desc "Driving in fog" \
--save_dir result_fog \
--batch_size 40

python3 main.py \
--dataset cityscapes \
--data_root /workspace/cityscapes \
--ckpt checkpoint/CS_source.pth \
--batch_size 30 \
--lr 0.01 \
--ckpts_path adapted_night \
--freeze_BB \
--train_aug \
--total_itrs 2000 --path_mu_sig result_night

python3 main.py \
--dataset cityscapes \
--data_root /workspace/cityscapes \
--ckpt checkpoint/CS_source.pth \
--batch_size 30 \
--lr 0.01 \
--ckpts_path adapted_snow \
--freeze_BB \
--train_aug \
--total_itrs 2000 --path_mu_sig result_snow

python3 main.py \
--dataset cityscapes \
--data_root /workspace/cityscapes \
--ckpt checkpoint/CS_source.pth \
--batch_size 30 \
--lr 0.01 \
--ckpts_path adapted_rain \
--freeze_BB \
--train_aug \
--total_itrs 2000 \
--mix \
--path_mu_sig result_rain

python3 main.py \
--dataset cityscapes \
--data_root /workspace/cityscapes \
--ckpt checkpoint/CS_source.pth \
--batch_size 30 \
--lr 0.01 \
--ckpts_path adapted_fog \
--freeze_BB \
--train_aug \
--total_itrs 2000 --path_mu_sig result_fog
