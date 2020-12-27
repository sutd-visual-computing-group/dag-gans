We use implement our DAG on its BigGAN and StyleGAN2 backbones to fairly compare to the concurrent work of Data-Efficient GANs based on its published [code](https://github.com/mit-han-lab/data-efficient-gans). 

We share our BigGAN + DAG code. Simply substituting these files in the original code of Data-Efficient GANs. We also provide the script examples to run our DAG. Here, we highlight our performance when reproducing the results (best IS and FID) the code of Data-Efficient GAN and compare to this work:

| Model name                               | Dataset           | is10k     | fid10k    |
| -----------------------------------------| ------------------| --------- | --------- |
| BigGAN                                   | `C10` (10%)       | 7.03      | 48.3      |
| BigGAN-DiffAugment (translation + cutout)| `C10` (10%)       | 8.40      | 23.9      |
| BigGAN-DAG (rotation + cropping)         | `C10` (10%)       | 8.63      | 23.6      |
| BigGAN-DAG (rotation + cropping + translation + cutout)         | `C10` (10%)       | 8.65      | 21.3      |
| BigGAN                                   | `C10` (20%)       | 8.51      | 22.3      |
| BigGAN-DiffAugment (translation + cutout)| `C10` (20%)       | 8.79      | 14.6      |
| BigGAN-DAG (rotation + cropping)         | `C10` (20%)       | 8.83      | 14.1      |
| BigGAN-DAG (rotation + cropping + translation + cutout)         | `C10` (20%)       | 8.84      | 13.5      |

In the above results, we demonstrate that our model can be applied to any data augmentations. For example, when combining our (rotation + cropping) and (translation + cutout) of DiffAugment, DAG can improve the baselines more significantly. To further improve the performance, you can simply add more data augmentation techniques you need. 


