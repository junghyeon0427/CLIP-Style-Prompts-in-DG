python3 main.py \
--dataset ACDC \
--data_root /workspace/dataset/ACDC \
--ckpt adapted_gtav/adapted_deeplabv3plus_resnet_clip_gta5.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub night

python3 main.py \
--dataset ACDC \
--data_root /workspace/dataset/ACDC \
--ckpt adapted_gtav/adapted_deeplabv3plus_resnet_clip_gta5.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub rain

python3 main.py \
--dataset ACDC \
--data_root /workspace/dataset/ACDC \
--ckpt adapted_gtav/adapted_deeplabv3plus_resnet_clip_gta5.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub snow

python3 main.py \
--dataset ACDC \
--data_root /workspace/dataset/ACDC \
--ckpt adapted_gtav/adapted_deeplabv3plus_resnet_clip_gta5.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub fog
