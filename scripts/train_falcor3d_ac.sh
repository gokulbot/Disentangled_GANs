#!/usr/bin/env bash
cd ..
python main.py --dataset falcor3d_7 --img_size 1024 --num_channels 3 --gpu_num 8 --progressive True --phase train \
--labels_fine --labels_coarse 0,1,2,3,4,5,6 --inp_res 4 --input_layer_type const --D_mode separate \
--cond_weight 10.0 --cond_type L2 --use_z True --use_noise False --use_instance_norm True --resume_snapshot -1 \
--batch_size_test 4 --iteration 1280 --max_iteration 25600 --labels_keep_rate 1.0 --style_res 2048 --seed 1 \
--FactorVAE False --FID True --MIG True --L2 True --use_style_mod True
