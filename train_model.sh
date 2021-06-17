#!/bin/sh

# python3 tools/train.py myConfigs/horse/train_spine_resnet_black.py --work-dir training_result/horse_spine_black/resnet
# python3 tools/train.py myConfigs/horse/train_spine_resnet_green.py --work-dir training_result/horse_spine_green/resnet
# python3 tools/train.py myConfigs/horse/train_spine_resnet_grey.py --work-dir training_result/horse_spine_grey/resnet
# python3 tools/train.py myConfigs/horse/train_spine_resnet_white.py --work-dir training_result/horse_spine_white/resnet

# python3 tools/train.py myConfigs/horse/train_leg_front_resnet_black.py --work-dir training_result/horse_leg_front_black/resnet
# python3 tools/train.py myConfigs/horse/train_leg_front_resnet_green.py --work-dir training_result/horse_leg_front_green/resnet
# python3 tools/train.py myConfigs/horse/train_leg_front_resnet_grey.py --work-dir training_result/horse_leg_front_grey/resnet
# python3 tools/train.py myConfigs/horse/train_leg_front_resnet_white.py --work-dir training_result/horse_leg_front_white/resnet

# python3 tools/train.py myConfigs/horse/train_leg_back_resnet_black.py --work-dir training_result/horse_leg_back_black/resnet
# python3 tools/train.py myConfigs/horse/train_leg_back_resnet_green.py --work-dir training_result/horse_leg_back_green/resnet
# python3 tools/train.py myConfigs/horse/train_leg_back_resnet_grey.py --work-dir training_result/horse_leg_back_grey/resnet

#python3 tools/train.py myConfigs/horse/train_leg_back_resnet_white.py --work-dir training_result/horse_leg_back_white/resnet
#python3 tools/train.py my_configs/cattle/resnet_50/head_black.py --work-dir training_result/cattle/resnet_50/head_black/


for animal in 'cattle'
do
  for model in 'resnet_50' 'resnet_150'
  do
    for color in 'black' 'green' 'grey' 'white'
    do
      for part in 'head' 'tail' 'spine' 'leg_front' 'leg_back'
      do
        echo my_configs/$animal/$model/$part'_'$color.py train_result/$animal/$model/$part_$color >> train_model.txt
        python3 tools/train.py my_configs/$animal/$model/$part'_'$color.py --work-dir train_result/$animal/$model/$part_$color
      done
    done
  done
done




