#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#ckpt_id=$2
images_path=/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/misc/test_image/
#images_path=/home/chris/Documents/PROJECTS/OXIT-Sport_Framework/outputs/dataset/

### DEFINE THE ROOT PATH HERE ###
#abs_root=/home/chris/Documents/PROJECTS/CLIP4STR

exp_path=/home/chris/Documents/PROJECTS/CLIP4STR/output/vl4str_large_5epoch_v2_2025-01-07_10-33-57/checkpoints/last.ckpt
#exp_path=/home/chris/Documents/PROJECTS/CLIP4STR/pretrained/clip/clip4str_huge_5eef9f86e2.pt
runfile=../read.py


python ${runfile} ${exp_path} --images_path ${images_path}
