import os
from argparse import ArgumentParser

import cv2
import numpy as np

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
import json


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    print(args)
    assert args.show or (args.out_video_root != '')
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    print('dataset', dataset)
    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Faild to load video file {args.video_path}'
    print('cap', cap)
    fps = cap.get(5)
    #fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps', fps)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('frame size', size)
    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    pose_res_json = {'result': []}
    pose_res_list = []
    img_idx = 0
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break
        # current frame image pose result
        temp_img_pose = {'id': img_idx, 'keypoints': []}
        # keep the person class bounding boxes.
        person_results = [{'bbox': np.array([0, 0, size[0], size[1]])}]

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            format='xyxy',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        #pose_results contains
        #print('pose_results', pose_results[0])
        #print('******')
        #print('returned_outputs', returned_outputs)
        # update data
        temp_img_pose['id'] = img_idx
        temp_img_pose['keypoints'] = pose_results[0]['keypoints'].tolist()
        if img_idx % 100 == 0:
            print('complete ', img_idx, ' frames.')
        img_idx+=1

        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)
        # save current pose result
        pose_res_list.append(temp_img_pose)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()
    # update pose result list
    print('frames count: ', len(pose_res_list))
    pose_res_json['result'] = pose_res_list
    # save json data
    with open('./video_leg_front.json', 'w') as json_file:
        json.dump(pose_res_json, json_file)


if __name__ == '__main__':
    main()
