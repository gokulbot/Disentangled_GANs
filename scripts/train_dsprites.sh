#!/usr/bin/env bash
cd ..
python3 main.py --dataset dsprites --img_size 64 --num_channels 1 --gpu_num 8 --progressive True --phase train \
--labels_fine --labels_coarse 0,1,2,3,4 --inp_res 4 --input_layer_type const --D_mode separate \
--cond_weight 1.0 --cond_type L2 --use_z True --use_noise False --use_instance_norm True --resume_snapshot -1 \
--batch_size_test 4 --iteration 2220 --max_iteration 44400 --labels_keep_rate 1.0 --style_res 128 --seed 1 \
--FactorVAE True --FID True --MIG True --L2 True --use_style_mod True