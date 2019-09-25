#!/usr/bin/env bash
cd ..
python main.py --dataset ffhq --img_size 1024 --num_channels 3 --gpu_num 8 --progressive True --phase train \
--inp_res 4 --input_layer_type const --D_mode together --seed 1 \
--cond_weight 1.0 --cond_type L2 --use_z True --use_noise False --use_instance_norm True --resume_snapshot -1 \
--batch_size_test 4 --FactorVAE False --FID True --MIG False --L2 False --use_style_mod True