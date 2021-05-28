import os

import cv2
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import numpy as np
import torch as tr
from torchvision import transforms
from mmpose.apis import (inference, inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.models import build_posenet
from mmpose.datasets.pipelines import Compose
import PIL
from PIL import Image


class LoadImage:
    """A simple pipeline to load image."""

    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the img_or_path.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
            img = mmcv.imread(results['img_or_path'], self.color_type,
                              self.channel_order)
        elif isinstance(results['img_or_path'], np.ndarray):
            results['image_file'] = ''
            if self.color_type == 'color' and self.channel_order == 'rgb':
                img = cv2.cvtColor(results['img_or_path'], cv2.COLOR_BGR2RGB)
        else:
            raise TypeError('"img_or_path" must be a numpy array or a str or '
                            'a pathlib.Path object')
        results['img'] = img
        return results


def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location=device)
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def vis_pose(img, points):
    points = points[0]
    for i, point in enumerate(points):
        x, y, p = point
        x = int(x)
        y = int(y)
        cv2.circle(img, (x, y), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(img, '{}'.format(i), (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)
    return img

def _box2cs(cfg, box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale

def process_model(model,  dataset,person_results, img_or_path):
    bboxes = np.array([box['bbox'] for box in person_results])
    cfg = model.cfg
    flip_pairs = None
    device = next(model.parameters()).device
    channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    if dataset in ('TopDownCocoDataset', 'TopDownOCHumanDataset',
                   'AnimalMacaqueDataset'):
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                      [13, 14], [15, 16]]
    elif dataset == 'TopDownCocoWholeBodyDataset':
        body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                [15, 16]]
        foot = [[17, 20], [18, 21], [19, 22]]

        face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

        hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                [111, 132]]
        flip_pairs = body + foot + face + hand
    elif dataset == 'TopDownAicDataset':
        flip_pairs = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
    elif dataset == 'TopDownMpiiDataset':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    elif dataset == 'TopDownMpiiTrbDataset':
        flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7],
                      [8, 9], [10, 11], [14, 15], [16, 22], [28, 34], [17, 23],
                      [29, 35], [18, 24], [30, 36], [19, 25], [31,
                                                               37], [20, 26],
                      [32, 38], [21, 27], [33, 39]]
    elif dataset in ('OneHand10KDataset', 'FreiHandDataset', 'PanopticDataset',
                     'InterHand2DDataset'):
        flip_pairs = []
    elif dataset == 'Face300WDataset':
        flip_pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11],
                      [6, 10], [7, 9], [17, 26], [18, 25], [19, 24], [20, 23],
                      [21, 22], [31, 35], [32, 34], [36, 45], [37,
                                                               44], [38, 43],
                      [39, 42], [40, 47], [41, 46], [48, 54], [49,
                                                               53], [50, 52],
                      [61, 63], [60, 64], [67, 65], [58, 56], [59, 55]]
    elif dataset == 'FaceAFLWDataset':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9],
                      [12, 14], [15, 17]]

    elif dataset == 'FaceCOFWDataset':
        flip_pairs = [[0, 1], [4, 6], [2, 3], [5, 7], [8, 9], [10, 11],
                      [12, 14], [16, 17], [13, 15], [18, 19], [22, 23]]

    elif dataset == 'FaceWFLWDataset':
        flip_pairs = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27],
                      [6, 26], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21],
                      [12, 20], [13, 19], [14, 18], [15, 17], [33,
                                                               46], [34, 45],
                      [35, 44], [36, 43], [37, 42], [38, 50], [39,
                                                               49], [40, 48],
                      [41, 47], [60, 72], [61, 71], [62, 70], [63,
                                                               69], [64, 68],
                      [65, 75], [66, 74], [67, 73], [55, 59], [56,
                                                               58], [76, 82],
                      [77, 81], [78, 80], [87, 83], [86, 84], [88, 92],
                      [89, 91], [95, 93], [96, 97]]

    elif dataset == 'AnimalFlyDataset':
        flip_pairs = [[1, 2], [6, 18], [7, 19], [8, 20], [9, 21], [10, 22],
                      [11, 23], [12, 24], [13, 25], [14, 26], [15, 27],
                      [16, 28], [17, 29], [30, 31]]
    elif dataset == 'AnimalHorse10Dataset':
        flip_pairs = []

    elif dataset == 'AnimalLocustDataset':
        flip_pairs = [[5, 20], [6, 21], [7, 22], [8, 23], [9, 24], [10, 25],
                      [11, 26], [12, 27], [13, 28], [14, 29], [15, 30],
                      [16, 31], [17, 32], [18, 33], [19, 34]]

    elif dataset == 'AnimalZebraDataset':
        flip_pairs = [[3, 4], [5, 6]]

    elif dataset == 'AnimalPoseDataset':
        flip_pairs = [[0, 1], [2, 3], [8, 9], [10, 11], [12, 13], [14, 15],
                      [16, 17], [18, 19]]
    else:
        raise NotImplementedError()
    batch_data = []
    for bbox in bboxes:
        center, scale = _box2cs(cfg, bbox)
        # prepare data
        data = {
            'img_or_path':
            img_or_path,
            'center':
            center,
            'scale':
            scale,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            dataset,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs
            }
        }
        data = test_pipeline(data)
        batch_data.append(data)
    batch_data = collate(batch_data, samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter not work so just move image to cuda device
        batch_data['img'] = batch_data['img'].to(device)
    # get all img_metas of each bounding box
    batch_data['img_metas'] = [
        img_metas[0] for img_metas in batch_data['img_metas'].data
    ]
    #torch_data = tr.tensor(input_data)
    #tran = transforms.ToTensor()
    #torch_data = tran(input_data).unsqueeze(0)
    #torch_data = torch_data.to(device)
    #print(batch_data['img'])
    #print(" ")
    #print(input_data)
    #print(" ")
    #print(torch_data - batch_data['img'])
    with tr.no_grad():
        result = model(
            img=batch_data['img'],
            #img = torch_data,
            img_metas=batch_data['img_metas'],
            return_loss=False,
            return_heatmap=False)
    return result['preds'], result['output_heatmap']



# example script:

device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")


model_head = init_pose_model(config='train_head_resnet.py', checkpoint='epoch_210.pth', device = device)
model_spine = init_pose_model(config='train_spine_resnet.py', checkpoint='spine_best.pth', device = device)

dataset_head = model_head.cfg.data['test']['type']
dataset_spine = model_spine.cfg.data['test']['type']


#poses, heatmap = _inference_single_pose_model(model, img_or_path, bboxes_xywh, dataset, return_heatmap=return_heatmap)

#image = Image.open('test_one_cow.jpg')
#data = np.asarray(image)
#data = np.expand_dims(data, axis=0)
#data = np.transpose(data, [0,3,1,2])
#trdata = torch.from_numpy(data).float().to(device)

image = cv2.imread('test_one_cow.jpg')
#data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#data = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(256, 256), mean=[0.485, 0.456, 0.406], swapRB=True, crop=False)
#data = mmcv.imread('test_0.png', 'color', 'rgb')

#img = cv2.cvtColor('test_0.png', cv2.COLOR_BGR2RGB)

# 'name': 'Head', 'left': 508.40299129486084, 'top': 218.73074993491173, 'width': 135.03000140190125, 'height': 116.78100228309631
head_box_1 = [508.40299129486084, 218.73074993491173, 135.03000140190125, 116.78100228309631]
head_result = []
head_result.append({'bbox': head_box_1})

preds, _ = process_model(model_head, dataset_head, head_result, image)
img = vis_pose(image, preds)

cow_box_1 = [234.15699899196625, 222.9727528989315, 411.599999666214, 267.4979969859123]
body_result = []
body_result.append({'bbox': cow_box_1})

preds2, _ = process_model(model_spine, dataset_spine, body_result, image)
img = vis_pose(img, preds2)

for bbox in [head_box_1, cow_box_1]:
    x,y,w,h = bbox
    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)

cv2.imshow('image',img)
cv2.waitKey(0)



# test a single image, with a list of bboxes.
'''
pose_results, _ = inference_top_down_pose_model(
        model, 'test_0.png', person_result, format='xywh', dataset='AnimalHorse10Dataset')
'''

#_______________________________________________________________



#_______________________________________________________________
#print(pose_results)

#vis_pose_result(model, 'test_0.png', pose_results)

#res = model(trdata)
#print(res)
