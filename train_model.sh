#!/bin/sh

for animal in 'cattle'
do
  for model in 'resnet_50'
  do
    for color in 'grey' 'black'
    do
      for part in 'head' 'spine' 'tail' 'leg_front' 'leg_back'
      do
        echo my_configs/$animal/$model'_10'/$part'_'$color.py train_result/$animal/$model'_10'/$part'_'$color/ >> train_model.txt
        python3 tools/train.py my_configs/$animal/$model'_10'/$part'_'$color.py --work-dir train_result/$animal/$model'_10'/$part'_'$color/
      done
    done
  done
done




