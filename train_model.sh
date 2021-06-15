#!/bin/sh

<<<<<<< HEAD
python3 tools/train.py myConfigs/horse/train_head_resnet.py --work-dir training_result/horse_head_black/resnet

python3 tools/train.py myConfigs/horse/train_tail_resnet.py --work-dir training_result/horse_tail_black/resnet
=======
#python3 tools/train.py myConfigs/horse/train_leg_front_resnet.py --work-dir temp_logs/horse_leg_front/resnet

#python3 tools/train.py myConfigs/horse/train_leg_back_resnet.py --work-dir temp_logs/horse_leg_back/resnet

#python3 tools/train.py myConfigs/horse/train_head_hrnet.py --work-dir temp_logs/horse_head/hrnet

#python3 tools/train.py myConfigs/horse/train_tail_hrnet.py --work-dir temp_logs/horse_tail/hrnet

#python3 tools/train.py myConfigs/horse/train_leg_front_hrnet.py --work-dir temp_logs/horse_leg_front/hrnet

#python3 tools/train.py myConfigs/horse/train_leg_back_hrnet.py --work-dir temp_logs/horse_leg_back/hrnet
python3 tools/train.py myConfigs/horse/train_head_right_resnet.py --work-dir temp_logs/horse_head_right/resnet

python3 tools/train.py myConfigs/horse/train_head_left_resnet.py --work-dir temp_logs/horse_head_left/resnet

>>>>>>> parent of 7c56a2c... update cofigs
