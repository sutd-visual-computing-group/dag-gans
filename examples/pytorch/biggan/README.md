We use implement our DAG on its BigGAN and StyleGAN2 backbones to fairly compare to the concurrent work of Data-Efficient GANs using its published code: https://github.com/mit-han-lab/data-efficient-gans 

We share our BigGAN + DAG can be replaced in the original code of Data-Efficient GANs paper. Here, we highlight our performance when we reproduce the results (best IS and FID) from the code and compare to Data-Efficient GANs:

| Model name                               | Dataset           | is10k     | fid10k    |
| -----------------------------------------| ------------------| --------- | --------- |
| BigGAN                                   | `C10` (10%)       | 7.03      | 48.3      |
| BigGAN-DiffAugment (translation + cutout)| `C10` (10%)       | 8.40      | 23.9      |
| BigGAN-DAG (rotation + cropping)         | `C10` (10%)       | 8.63      | 23.6      |
| BigGAN-DAG (fliprot + cropping)          | `C10` (10%)       | 8.50      | 23.9      |
| BigGAN                                   | `C10` (20%)       | 0.00      | 0.00      |
| BigGAN-DiffAugment (translation + cutout)| `C10` (20%)       | 8.79      | 14.6      |
| BigGAN-DAG (rotation + cropping)         | `C10` (20%)       | 8.83      | 14.1      |
| BigGAN-DAG (fliprot + cropping)          | `C10` (20%)       | 0.00      | 0.00      |

Note that here we just report the augmetnations of fliprot and cropping as mentioned in our paper. However, our model is not limited to the number of augmentations to be used. To futher improve the performance, you can simply add more data augmentation techniques. 
