# Crowd Counting Baseline

We set up a strong baseline for crowd counting easy to follow and implement. Following tricks are adopted to improve the performance. We are continuing with this work.

Experiments are conducted on ShanghaiTech PartA dataset.

## Baseline Network Architecture

First 13 layers of VGG-16 without batch-norm followed by upsample and conv layers to get **the same size** density maps as input images. In the backend, SE blocks and Swish activation are adopted. Figure will be added soon.



## Augmentation

| Strategy     |      MAE |     RMSE |      PSNR |       SSIM | PSNR(1/8) | SSIM(1/8) |       Time/epoch |
| ------------ | -------: | -------: | --------: | ---------: | --------: | --------: | ---------------: |
| 0.3$\times$  | **62.9** |     99.7 | **58.61** | **0.9869** | **22.51** |      0.62 | **0.33$\times$** |
| 0.4$\times$  |     64.8 | **95.8** |     58.44 |     0.9864 |     22.35 |      0.61 |     0.39$\times$ |
| 0.5$\times$  |     64.3 |    100.4 |     58.27 |     0.9861 |     22.18 |      0.58 |     0.43$\times$ |
| 0.6$\times$  |     64.4 |     98.8 |     58.36 |     0.9862 |     22.27 |      0.60 |     0.51$\times$ |
| 0.7$\times$  |     64.8 |     99.7 |     58.16 |     0.9858 |     22.09 |      0.56 |     0.61$\times$ |
| 0.8$\times$  |     64.2 |     99.6 |     58.23 |     0.9858 |     22.15 |      0.60 |     0.71$\times$ |
| 0.9$\times$  |     66.5 |    100.3 |     58.12 |     0.9856 |     22.04 |      0.58 |     0.86$\times$ |
| Original     |     67.7 |    103.1 |     58.02 |     0.9854 |     21.94 |      0.56 |     1.00$\times$ |
| fixed        |     64.9 |    101.2 |     58.34 |     0.9862 |     22.26 |      0.62 |     1.20$\times$ |
| fixed+random |     63.8 |    101.1 |     58.39 |     0.9865 |     22.30 |  **0.64** |     2.43$\times$ |
| mixed        |     68.7 |    106.5 |     58.01 |     0.9854 |     21.92 |      0.56 |     5.01$\times$ |



## Map Size

|               |  MAE | RMSE | PSNR | SSIM | PSNR(1/8) | SSIM(1/8) |
| ------------- | ---: | ---: | ---: | ---: | --------: | --------: |
| 1             | 62.9 | 99.7 |      |      |           |           |
| $\frac{1}{2}$ |      |      |      |      |           |           |
| $\frac{1}{4}$ |      |      |      |      |           |           |
| $\frac{1}{8}$ |      |      |      |      |           |           |



## Loss Function

|                 |  MAE | RMSE | PSNR | SSIM |
| --------------- | ---: | ---: | ---: | ---: |
| $L_{MSE}$       |      |      |      |      |
| $L_{MSE}+L_{C}$ |      |      |      |      |
| $MSSIM$         |      |      |      |      |
|                 |      |      |      |      |



## Learning Objective

|                                  |  MAE | RMSE | PSNR | SSIM |
| -------------------------------- | ---: | ---: | ---: | ---: |
| Density Map                      |      |      |      |      |
| Density Map + Soft Attention Map |      |      |      |      |
| Density Map + Hard Attention Map |      |      |      |      |



