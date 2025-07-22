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
import time

import misc.craft.imgproc as imgproc
from torch.autograd import Variable
import misc.craft.craft_utils as craft_utils


@torch.inference_mode()

def visualization(frame, boxes, character_bboxes):
    
    if len(boxes) != 0:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        for idx in range(len(boxes)):
            cv2.rectangle(frame, (int(boxes[idx][0]),int(boxes[idx][3])), (int(boxes[idx][2]),int(boxes[idx][1])), (0, 255, 0) , 2)
            if len(character_bboxes[idx]) != 0:
                    cv2.rectangle(frame, (int(boxes[idx][0]),int(boxes[idx][3])), (int(boxes[idx][2]),int(boxes[idx][1])), (0, 0, 255) , 2)
            
        cv2.imshow("test", frame)
        
        cv2.waitKey(1)
        
        # key_pressed = False
        # while not key_pressed:
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord("q"):
        #             key_pressed = True

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
    
def get_character_position(img, brightness_thresh, output_img_=None):

    bb_size_threshold = 10  ### Minimum size of the bounding box ###
    color_threshold = 0.3   ### Brightness value, that determines the lower threshold of the color palette; lower value - red , upper value - green ###

    ret, thresh = cv2.threshold(img, brightness_thresh, 1, 0)
    thresh = cv2.convertScaleAbs(thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:

        #find the biggest area of the contour
        c = max(contours, key = cv2.contourArea)

        if output_img_ is not None:

            # the contours are drawn here
            #cv.drawContours(output_img_, [c], -1, 255, 3)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img)
            detection_color = (0, 255*(maxVal-color_threshold), 255-255*(maxVal-color_threshold))
            cv2.circle(output_img_, maxLoc, 1, detection_color, 10)
            cv2.circle(output_img_, maxLoc, 20, detection_color, 2)

        x,y,w,h = cv2.boundingRect(c)
        return [x+w/2, y+h/2]#, thresh
    
def get_craft_result( net, image, text_threshold, link_threshold, low_text, canvas_size, mag_ratio, max_intensity, show_time, cuda, poly, refine_net=None):
    
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        input_name = net.get_inputs()[0].name
        output_names = [o.name for o in net.get_outputs()] 
        
        x_numpy = x.detach().cpu().numpy().astype(np.float32)
        
        outputs = net.run(output_names, {input_name: x_numpy})
        
        y = outputs[0]
        feature = outputs[0]

    # make score and link map
    score_text = y[0,:,:,0]
    score_link = y[0,:,:,1]

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    if max_intensity == True:
        result_img = score_text.copy()
        result_pos = get_character_position(score_text, 0.1, result_img)
        result_img = cv2.normalize(result_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        #cv2.imwrite('test_heatmap.jpg', result_test)
        return result_pos, result_img
    else:
        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes, polys, ret_score_text
    
def forward_craft(onnx_session, frame, boxes) -> None:
    #parameters = cfg.models.craft.parameters
    
    text_threshold = 0.4
    low_text = 0.2
    link_threshold = 0.4
    canvas_size = 1280
    mag_ratio = 10

    outputs = []

    for bbox in boxes:

        #print(bbox)

        if all(i >= 0 for i in bbox):
            cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            _bboxes, polys, score_text = get_craft_result(onnx_session, cropped_image, text_threshold, link_threshold, low_text, canvas_size, mag_ratio, False, False, True, False, None)

            outputs.append(polys)
        else:
            outputs.append([])

        """ cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 0, 255) , 2) 
        cv2.imshow("test", img)
        cv2.imshow("test", cropped_image) """

    return outputs

def forward_clip4str(frame):
    device= "cuda"
    images_path = "/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/misc/test_image/"
    checkpoint = "/home/chris/Documents/PROJECTS/CLIP4STR/output/vl4str_large_5epoch_v2_2025-01-07_10-33-57/checkpoints/last.ckpt"

    
    

def main_with_visualization():
    
    rt_detr_session = onnxruntime.InferenceSession("code/CLIP4STR/misc/rtdetrv2.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    
    craft_session = onnxruntime.InferenceSession("code/CLIP4STR/misc/craft/craft.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    
    video_cap = cv2.VideoCapture("/home/chris/Documents/VIDEOS_FOR_ANNOTATION/CUT_VIDEOS/cut_5_0.mp4")
    
    while True:
        has_frame, frame = video_cap.read()
        if not has_frame:
            break
        
        frame_original = frame.copy()
        frame = torch.from_numpy(frame).unsqueeze(0).to('cuda')
        
        labels, boxes, scores = forward_rtdetr(rt_detr_session, frame, (1920, 1080))
        
        character_bboxes = forward_craft(craft_session, frame_original, boxes)
        
        character_list = forward_clip4str(frame_original, boxes, character_bboxes)
        
        visualization(frame_original, boxes, character_bboxes)

def forward_clip4str(frame_original, boxes, character_bboxes):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    # parser.add_argument('--images_path', type=str, help='Images to read')
    # parser.add_argument('--device', default='cuda')
    # args, unknown = parser.parse_known_args()
    # kwargs = parse_model_args(unknown)
    # print('KWARGS: ',kwargs)
    # print(f'Additional keyword arguments: {kwargs}')
    
    device= "cuda"
    #images_path = "/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/misc/test_image/"
    checkpoint = "/home/chris/Documents/PROJECTS/CLIP4STR/output/vl4str_large_5epoch_v2_2025-01-07_10-33-57/checkpoints/last.ckpt"

    model = load_from_checkpoint(checkpoint).eval().to(device)
    
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    
    print(model.hparams.img_size)

    #files = sorted([x for x in os.listdir(images_path) if x.endswith('png') or x.endswith('jpeg') or x.endswith('jpg')])
    
    outputs = []

    for idx, bbox in enumerate(boxes):

        cropped_img = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        current_craft_result = character_bboxes[idx]
        if len(current_craft_result) !=0:

            possible_numbers = [0,1,2,3,4,5,6,7,8,9]

            max_idx = np.argmax(current_craft_result[0], axis=0)
            min_idx = np.argmin(current_craft_result[0], axis=0)
            max_x, max_y = current_craft_result[0][max_idx]
            min_x, min_y = current_craft_result[0][min_idx]

            maxX = max_x[0]
            minX = min_x[0]
            maxY = max_y[1]
            minY = min_y[1]

            _img = Image.fromarray(cropped_img.astype('uint8'), 'RGB')

            crop_img = _img.crop((minX, minY, maxX, maxY))

            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            # cv2.rectangle(cropped_img, (int(minX),int(maxY)), (int(maxX),int(minY)), (0, 0, 255) , 2)
            # cv2.imshow("test", cropped_img)

            # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
            _img = img_transform(crop_img).unsqueeze(0)

            p = model(frame_original).softmax(-1)
            pred, p = model.tokenizer.decode(p)
            print(pred[0])

    return outputs

    # for box in character_bboxes:
    #     # Load image and prepare for input
    #     #filename = os.path.join(images_path, fname)
    #     #image = Image.open(filename).convert('RGB')
    #     image = 
    #     #print(image.shape)
    #     image = img_transform(image).unsqueeze(0).to(device)
    #     #print(image.shape)
        
    #     #torch.onnx.export(model, image.to(args.device), "/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/scripts/clip4str.onnx", verbose=True)
    #     #model.to_onnx("/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/scripts/clip4str.onnx", image, export_params=True)
    #     #break

    #     p = model(image).softmax(-1)
    #     pred, p = model.tokenizer.decode(p)
    #     #print(f'{fname}: {pred[0]}')
    #     print(pred[0])


if __name__ == '__main__':
    #main()
    main_with_visualization()
