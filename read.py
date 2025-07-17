#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse

import torch

from PIL import Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

import onnxruntime
import cv2
import numpy as np
import copy


@torch.inference_mode()

def visualization(frame, boxes):
    
    if len(boxes) != 0:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        for idx in range(len(boxes)):
            cv2.rectangle(frame, (int(boxes[idx][0]),int(boxes[idx][3])), (int(boxes[idx][2]),int(boxes[idx][1])), (0, 255, 0) , 2)
            
        cv2.imshow("test", frame)
        
        cv2.waitKey(1)
        
        key_pressed = False
        while not key_pressed:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                    key_pressed = True

def forward_rtdetr(onnx_session, frame, orig_size) -> None:
        
        orig_size = np.array([[orig_size[0], orig_size[1]]])
        
        s_frame = frame.squeeze(dim=0)
        
        cpu_tensor = s_frame.cpu()
        numpy_frame = cpu_tensor.numpy()
        
        img = cv2.resize(numpy_frame, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        
        labels, boxes, scores = onnx_session.run(
                output_names=None,
                input_feed={
                    'images': img,
                    "orig_target_sizes": orig_size,
                },
        )
        
        return labels, boxes, scores

def main_with_visualization():
    
    rt_detr_session = onnxruntime.InferenceSession("code/CLIP4STR/misc/rtdetrv2.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    
    video_cap = cv2.VideoCapture("/home/chris/Documents/VIDEOS_FOR_ANNOTATION/CUT_VIDEOS/new_cut_1.mp4")
    
    while True:
        has_frame, frame = video_cap.read()
        if not has_frame:
            break
        
        frame_original = frame.copy()
        frame = torch.from_numpy(frame).unsqueeze(0).to('cuda')
        
        labels, boxes, scores = forward_rtdetr(rt_detr_session, frame, (1920, 1080))
        
        visualization(frame_original, boxes)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    # parser.add_argument('--images_path', type=str, help='Images to read')
    # parser.add_argument('--device', default='cuda')
    # args, unknown = parser.parse_known_args()
    # kwargs = parse_model_args(unknown)
    # print('KWARGS: ',kwargs)
    # print(f'Additional keyword arguments: {kwargs}')
    
    device= "cuda"
    images_path = "/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/misc/test_image/"
    checkpoint = "/home/chris/Documents/PROJECTS/CLIP4STR/output/vl4str_large_5epoch_v2_2025-01-07_10-33-57/checkpoints/last.ckpt"

    model = load_from_checkpoint(checkpoint).eval().to(device)
    
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    
    print(model.hparams.img_size)

    files = sorted([x for x in os.listdir(images_path) if x.endswith('png') or x.endswith('jpeg') or x.endswith('jpg')])

    for fname in files:
        # Load image and prepare for input
        filename = os.path.join(images_path, fname)
        image = Image.open(filename).convert('RGB')
        #print(image.shape)
        image = img_transform(image).unsqueeze(0).to(device)
        #print(image.shape)
        
        #torch.onnx.export(model, image.to(args.device), "/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/scripts/clip4str.onnx", verbose=True)
        #model.to_onnx("/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/scripts/clip4str.onnx", image, export_params=True)
        #break

        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        print(f'{fname}: {pred[0]}')


if __name__ == '__main__':
    #main()
    main_with_visualization()
