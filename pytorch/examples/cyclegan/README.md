We provide an example that DAG can improve image-image translation with CycleGAN.

The implementation based on [Pytorch code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Substituting the original code by our provided DAG code in the correct location and running the following command lines, e.g., on maps dataset:

```
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan_dag
```


| Model name                               | Per-pixel  acc.   | Per-class  acc.  | Class  IOU |
| -----------------------------------------| ------------------| ---------------- | ---------- |
| CycleGAN                                 | 0.52              | 0.17             | 0.11       |
| CycleGAN (pytorch)                       | 0.21              | 0.06             | 0.02       |
| CycleGAN (pytorch) + DAG                 | 0.59              | 0.19             | 0.15       |


