import os

import cv2
import mmcv
import numpy as np
import torch
from mmpose.apis import (inference, inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
import PIL
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = init_pose_model(config='train_head_resnet.py', checkpoint='epoch_210.pth', device = device)
image = Image.open('test_0.png')
data = np.asarray(image)
data = np.expand_dims(data, axis=0)
data = np.transpose(data, [0,3,1,2])
trdata = torch.from_numpy(data).float().to(device)

person_result = []
person_result.append({'bbox': [0, 0, 256, 256]})
# test a single image, with a list of bboxes.
pose_results, _ = inference_top_down_pose_model(
        model, 'test_0.png', person_result, format='xywh', dataset='AnimalHorse10Dataset')
print(pose_results)

#vis_pose_result(model, 'test_0.png', pose_results)

#res = model(trdata)
#print(res)
