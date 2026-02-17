#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#ckpt_id=$2
images_path=/home/aitrain/Documents/Krisztian/CLIP4STR/code/CLIP4STR/misc/test_image/

### DEFINE THE ROOT PATH HERE ###
#abs_root=/home/aitrain/Documents/Krisztian/CLIP4STR

exp_path=/home/aitrain/Documents/Krisztian/CLIP4STR/output/vl4str_2024-12-06_09-38-49/checkpoints/epoch_2.ckpt
runfile=../read.py


python ${runfile} ${exp_path} --images_path ${images_path}
