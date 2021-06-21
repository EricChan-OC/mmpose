#!/bin/sh

for animal in 'cattle'
do
  for model in 'resnet_50'
  do
    for color in 'black' 'grey'
    do
      for part in 'spine' 'leg_front' 'leg_back'
      do
        echo my_configs/$animal/$model'_5'/$part'_'$color.py train_result/$animal/$model'_5'/$part'_'$color/ >> train_model.txt
        python3 tools/train.py my_configs/$animal/$model'_5'/$part'_'$color.py --work-dir train_result/$animal/$model'_5'/$part'_'$color/





