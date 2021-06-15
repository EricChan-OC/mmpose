#!/bin/sh

python3 tools/train.py myConfigs/horse/train_head_resnet.py --work-dir training_result/horse_head_black/resnet

python3 tools/train.py myConfigs/horse/train_tail_resnet.py --work-dir training_result/horse_tail_black/resnet