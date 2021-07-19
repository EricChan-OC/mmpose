#!/bin/sh

for animal in 'horse'
do
  for model in 'resnet_50'
  do
    for color in 'grey' 'black'
    do
      for part in "head_left"
      do
      
          if [ $part = "head_front" ]
          then
            for extend in '0' '5' '10'
            do
            
              echo my_configs/$animal/$model'_'$extend/$part'_'$color.py train_result/$animal/$model'_'$extend/$part'_'$color/ >> train_model.txt
              python3 tools/train.py my_configs/$animal/$model'_'$extend/$part'_'$color.py --work-dir train_result_0623/$animal/$model'_'$extend/$part'_'$color/
            
            done
            
          elif [ $part = "head_left" ]
          then
            for extend in '0'
            do
            
              echo my_configs/$animal/$model'_'$extend/$part'_'$color.py train_result/$animal/$model'_'$extend/$part'_'$color/ >> train_model.txt
              python3 tools/train.py my_configs/$animal/$model'_'$extend/$part'_'$color.py --work-dir train_result_0623/$animal/$model'_'$extend/$part'_'$color/
            
            done
            
          elif [ $part = "head_right" ]
          then
            for extend in '0' '5' '10'
            do
            
              echo my_configs/$animal/$model'_'$extend/$part'_'$color.py train_result/$animal/$model'_'$extend/$part'_'$color/ >> train_model.txt
              python3 tools/train.py my_configs/$animal/$model'_'$extend/$part'_'$color.py --work-dir train_result_0623/$animal/$model'_'$extend/$part'_'$color/
            
            done
          
          elif [ $part = "spine" ]
          then
            for extend in '0' '5' '10'
            do
            
              echo my_configs/$animal/$model'_'$extend/$part'_'$color.py train_result/$animal/$model'_'$extend/$part'_'$color/ >> train_model.txt
              python3 tools/train.py my_configs/$animal/$model'_'$extend/$part'_'$color.py --work-dir train_result_0623/$animal/$model'_'$extend/$part'_'$color/
            
            done
          
          elif [ $part = "tail" ]
          then
            for extend in '20' '15'
            do
            
              echo my_configs/$animal/$model'_'$extend/$part'_'$color.py train_result/$animal/$model'_'$extend/$part'_'$color/ >> train_model.txt
              python3 tools/train.py my_configs/$animal/$model'_'$extend/$part'_'$color.py --work-dir train_result_0623/$animal/$model'_'$extend/$part'_'$color/
            
            done
          
          
          elif [ $part = "leg_front" ]
          then
            for extend in '15'
            do
            
              echo my_configs/$animal/$model'_'$extend/$part'_'$color.py train_result/$animal/$model'_'$extend/$part'_'$color/ >> train_model.txt
              python3 tools/train.py my_configs/$animal/$model'_'$extend/$part'_'$color.py --work-dir train_result_0623/$animal/$model'_'$extend/$part'_'$color/
            
            done
       
          
          elif [ $part = "leg_back" ]
          then
            for extend in '15'
            do
            
              echo my_configs/$animal/$model'_'$extend/$part'_'$color.py train_result/$animal/$model'_'$extend/$part'_'$color/ >> train_model.txt
              python3 tools/train.py my_configs/$animal/$model'_'$extend/$part'_'$color.py --work-dir train_result_0623/$animal/$model'_'$extend/$part'_'$color/
            
            done
          fi
            
      done
    done
  done
done




