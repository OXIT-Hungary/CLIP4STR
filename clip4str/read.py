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

import numpy as np

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

def main_init(checkpoint):
    
    model = load_from_checkpoint(checkpoint).eval().to('cuda')
    
    return model

def main_eval(checkpoint, image, character_pos, **kwargs):
    
    model = load_from_checkpoint(checkpoint).eval().to('cuda')
    
    for pos in character_pos:
        
        #Cropping from original image
        crop = image[round(pos[1]):round(pos[3]), round(pos[0]):round(pos[2])]
        #crop = image.crop((pos[0], pos[1], pos[2], pos[3]))

        #print(image.shape)
        #image = image[0]  # Remove batch dimension
        #print(image.shape)
        #image = np.transpose(image, (1, 2, 0))  # Convert from (3, width, height) to (width, height, 3)
        #print(image.shape)
        _image = Image.fromarray(crop.astype('uint8'), 'RGB')
        
        img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

        # Load image and prepare for input
        _image = img_transform(_image).unsqueeze(0).to('cuda')
        
        p = model(_image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        print('number: ', pred[0])

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images_path', type=str, help='Images to read')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(kwargs)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    files = sorted([x for x in os.listdir(args.images_path) if x.endswith('png') or x.endswith('jpeg') or x.endswith('jpg')])

    for fname in files:
        # Load image and prepare for input
        filename = os.path.join(args.images_path, fname)
        image = Image.open(filename).convert('RGB')
        image = img_transform(image).unsqueeze(0).to(args.device)

        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        print(f'{fname}: {pred[0]}')


if __name__ == '__main__':
    main()
