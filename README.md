Crowd Counting Baseline

We set up a strong baseline for crowd counting easy to follow and implement. Some tricks are adopted to improve the performance.

## Augmentation

| Strategy    | MAE  | RMSE | Speed     |
| ----------- | ---- | ---- | --------- |
| Original    |      |      | 1         |
| 0.3$\times$ |      |      | 2$\times$ |
| 0.4$\times$ |      |      |           |
| 0.5$\times$ |      |      |           |
| 0.6$\times$ |      |      |           |
| 0.7$\times$ |      |      |           |
| 0.8$\times$ |      |      |           |
| 0.9$\times$ |      |      |           |
|             |      |      |           |

## Loss Function

|                 | MAE  | RMSE |
| --------------- | ---- | ---- |
| $L_{MSE}$       |      |      |
| $L_{MSE}+L_{C}$ |      |      |
| $SSIM$          |      |      |



## Learning Objective

|                             | MAE  | RMSE |
| --------------------------- | ---- | ---- |
| Density Map                 |      |      |
| Density Map + Attention Map |      |      |

