# Crowd Counting Baseline

We set up a strong baseline for crowd counting easy to follow and implement. Following tricks are adopted to improve the performance. We are continuing with this work.

Experiments are conducted on ShanghaiTech PartA dataset.

## Baseline Network Architecture

First 13 layers of VGG-16 without batch-norm followed by upsample and conv layers to get **the same size** density maps as input images. In the backend, SE blocks is adopted. Swish activation replaces ReLU. Figure will be added soon.

Use full images and MSE loss function to train.

VGG16-13 not converge.

When applying pyramids in the whole decoder, the model converge to a bad local minimum, so only first 2 layers use pyramids.

Problems to be issued:

- [ ] When training U-VGG, loss converge to 1.1e-4. and MAE ~240. Try (1)use SGD (2)random init instead of vgg16 and simplify. but they do not work.

- [ ] When training CSRNet, if init the parameters randomly instead of using pretrained, converge to MAE ~330.
- [ ] If use first 13 layers of VGG16 in M_VGG, downsample = 8, training loss converges to ~0.0030 and do not reduce. MAE is 86.4.
- [x] When adding 3 upsample layers into CSRNet so that to regress 1 size density maps, loss converges to ~1.218e-5, and MAE > 200. When adding 2 upsample layers, , loss converges to ~1.8e-4, and MAE > 120. When adding 1 upsample layer, loss converges to ~2e-3, and MAE ~94. Reducing the lr will work.

| Model                                |   MAE |  RMSE |  PSNR | SSIM | Params |
| ------------------------------------ | ----: | ----: | ----: | ---: | -----: |
| ResNet-50 + decoder                  |  80.6 | 130.1 |       |      |        |
| InceptionV3 + decoder                | 119.4 | 170.5 |       |      |        |
| VGG16-10 + decoder(CSRNet)           |  71.0 | 111.7 | 22.66 | 0.70 |        |
| VGG16-10 + decoder + skip connection |   240 |       |       |      |        |
| VGG16-10 + decoder(1,3,5,7 filter)   |  70.9 | 110.8 |       |      |        |
| VGG16-10 + decoder(1,2,3,6 dilation) |  74.7 | 113.1 |       |      |        |
| VGG16-10 + decoder(depth pyramid)    |  74.2 | 112.5 |       |      |        |
| VGG16-13 + decoder                   |  86.4 | 125.1 |       |      |        |
| VGG16-10 + decoder, 1 size           |  >200 |  >300 |       |      |        |
| VGG16-10 + decoder, 1/2 size         | 119.4 | 192.9 |       |      |        |
| VGG16-10 + decoder, 1/4 size         |    94 |       |       |      |        |
| VGG16-13 + decoder, 1 size           |       |       |       |      |        |
| VGG16-13 + decoder, 1/2 size         |       |       |       |      |        |
| VGG16-13 + decoder, 1/4 size         |       |       |       |      |        |
| Dense                                |       |       |       |      |        |
| DenseRes                             |       |       |       |      |        |
| VGG16-13 + decoder + se              |       |       |       |      |        |
| VGG16-13 + decoder + se + swish      |       |       |       |      |        |

Select model VGG16-13 + decoder + se + swish



## Augmentation

When calculating PSNR and SSIM, different resolutions lead to different result. For original size density maps, the value of each pixel is quite small so that PSNR and SSIM is bigger than 1/8 size.

$Loss = L_{MSE}+100*downsample*L_{C}$

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

|                                              |      MAE |     RMSE |  PSNR |   SSIM | PSNR(1/8) | SSIM(1/8) |
| -------------------------------------------- | -------: | -------: | ----: | -----: | --------: | --------: |
| 1                                            |     62.9 |     99.7 | 58.61 | 0.9869 | **22.51** |  **0.62** |
| $\frac{1}{2}$                                | **62.0** |     95.4 | 46.47 | 0.9416 |     22.42 |  **0.62** |
| $\frac{1}{4}$                                | **62.0** | **93.0** | 34.38 | 0.8197 |     22.35 |      0.61 |
| $\frac{1}{4}, L_{MSE}+400*downsample*L_{C}$  |     61.4 |     92.6 |       |        |           |           |
| $\frac{1}{4}, L_{MSE}+1000*downsample*L_{C}$ |     60.0 |     92.6 |       |        |           |           |
| $\frac{1}{4}, L_{MSE}+25*downsample*L_{C}$   |     63.5 |     93.4 |       |        |           |           |
| $\frac{1}{8}$                                |     63.0 |     95.6 | 22.24 | 0.6122 |     22.24 |      0.61 |



## Loss Function

Size = 1

|                                |          MAE |  RMSE | PSNR | SSIM |
| ------------------------------ | -----------: | ----: | ---: | ---: |
| $L_{MSE}$                      |              |       |      |      |
| $L_{MSE}+100*downsample*L_{C}$ |         62.9 |  99.7 |      |      |
| $DMS-SSIM$                     |         71.7 | 108.9 |      |      |
| $MS-SSIM$                      |         69.8 | 104.4 |      |      |
| $L_{SA}+L_{SC}$                | Not converge |       |      |      |
| 1                              |         71.9 | 110.0 |      |      |
| 2                              |         70.6 | 112.9 |      |      |
| 3                              |         71.6 | 110.2 |      |      |
| 4                              |         69.9 | 109.2 |      |      |
| 5                              |         69.6 | 105.5 |      |      |



## Learning Objective

|                                  |  MAE | RMSE | PSNR | SSIM | Params(M) |
| -------------------------------- | ---: | ---: | ---: | ---: | --------: |
| Density Map                      |      |      |      |      |     23.45 |
| Density Map + Soft Attention Map |      |      |      |      |     23.53 |
| Density Map + Hard Attention Map |      |      |      |      |     23.53 |



