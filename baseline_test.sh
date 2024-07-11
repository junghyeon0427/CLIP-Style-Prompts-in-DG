# +
# test on night
# -

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_night/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub night

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_night/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub rain

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_night/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub snow

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_night/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub fog

# +
# test on rain
# -

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_rain/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub night

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_rain/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub rain

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_rain/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub snow

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_rain/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub fog

# +
# test on snow
# -

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_snow/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub night

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_snow/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub rain

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_snow/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub snow

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_snow/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub fog

# +
# test on fog
# -

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_fog/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub night

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_fog/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub rain

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_fog/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub snow

python3 main.py \
--dataset ACDC \
--data_root /workspace/ACDC \
--ckpt adapted_fog/adapted_deeplabv3plus_resnet_clip_cityscapes.pth \
--test_only \
--val_batch_size 10 \
--ACDC_sub fog
