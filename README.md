# The code for the paper "Disentangled GANs for Controllable Generation of High-Resolution Images"


## System requirements

* Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.10.0 or newer with GPU support.
* One or more high-end NVIDIA GPUs with at least 11GB of DRAM. We recommend NVIDIA DGX-1 with 8 Tesla V100 GPUs.
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.3.1 or newer.
* Other dependencies: tqdm 4.36.1, pillow 6.1.0, scikit-learn 0.20.2, scipy 0.13.3.

The above requirements are basically kept the same with those in the original StyleGAN implementation.

## Instructions
The `scripts` folders contain scripts for starting the different experiments.

* To reproduce the `AC-StyleGAN on Isaac3D` experiments, you can try:
```
cd scripts
bash train_isaac3d_ac.sh
```
or the `FC-StyleGAN on Isaac3D` experiments:
```
cd scripts
bash train_isaac3d_fc.sh
```

* Similarly, to reproduce the `AC-StyleGAN on Falcor3D` experiments, you can try:
```
cd scripts
bash train_falcor3d_ac.sh
```
or the `FC-StyleGAN on Isaac3D` experiments:
```
cd scripts
bash train_falcor3d_fc.sh
```

Note that in each script, `labels_keep_rate` represents the *fraction of labelled data* $\alpha$ in the paper and `cond_weight` denotes 
the *disentanglement coefficient* $\gamma$.

* To quickly evaluate the trained models, you can simply replace `--phase train` by `--phase eval` 
in each `train_xxxxxxx_xx.sh` and then run the command `bash train_xxxxxxx_xx.sh`.
