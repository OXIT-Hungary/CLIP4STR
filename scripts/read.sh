#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#ckpt_id=$2
images_path=/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/misc/test_image/

### DEFINE THE ROOT PATH HERE ###
#abs_root=/home/chris/Documents/PROJECTS/CLIP4STR

exp_path=/home/chris/Documents/PROJECTS/CLIP4STR/output/vl4str_2024-12-04_15-10-24/checkpoints/last.ckpt
runfile=../read.py


python ${runfile} ${exp_path} --images_path ${images_path}
