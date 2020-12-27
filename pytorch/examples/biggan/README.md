We use implement our DAG on its BigGAN and StyleGAN2 backbones to fairly compare to the concurrent work of Data-Efficient GANs using its published code: https://github.com/mit-han-lab/data-efficient-gans 

We share our BigGAN + DAG files can be replaced in the original code of Data-Efficient GANs. Here, we highlight our performance when reproducing the results (best IS and FID) the code of Data-Efficient GAN and compare to this work:

| Model name                               | Dataset           | is10k     | fid10k    |
| -----------------------------------------| ------------------| --------- | --------- |
| BigGAN                                   | `C10` (10%)       | 7.03      | 48.3      |
| BigGAN-DiffAugment (translation + cutout)| `C10` (10%)       | 8.40      | 23.9      |
| BigGAN-DAG (rotation + cropping)         | `C10` (10%)       | 8.63      | 23.6      |
| BigGAN-DAG (rotation + cropping + translation + cutout)         | `C10` (10%)       | 8.65      | 21.3      |
| BigGAN                                   | `C10` (20%)       | 8.51      | 22.3      |
| BigGAN-DiffAugment (translation + cutout)| `C10` (20%)       | 8.79      | 14.6      |
| BigGAN-DAG (rotation + cropping)         | `C10` (20%)       | 8.83      | 14.1      |

Note that our model is not limited to above augmentations. To futher improve the performance, you can simply add more data augmentation techniques you need. 
