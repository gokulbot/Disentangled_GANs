#!/usr/bin/env bash
cd ..
python main.py --dataset isaac3d --img_size 512 --num_channels 3 --gpu_num 8 --progressive True --phase train \
--labels_fine 4,5,7,8 --inp_res 16 --input_layer_type down --D_mode separate \
--cond_weight 1.0 --cond_type L2 --use_z True --use_noise False --use_instance_norm True --resume_snapshot -1 \
--batch_size_test 4 --iteration 1480 --max_iteration 29600 --labels_keep_rate 1.0 --style_res 16 --seed 1 \
--FactorVAE False --FID True --MIG True --L2 True --use_style_mod True
