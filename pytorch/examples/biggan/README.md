We implement our DAG on its BigGAN based on [DiffAugment](https://github.com/mit-han-lab/data-efficient-gans) for a fair comparision to this work. 

We update our BigGAN + DAG code on CIFAR-10 first. Simply substituting these files in the original code. We also provide the script examples to run our DAG. Here, we highlight our performance as compared to this work. Note that our model is trained on Single Titan RTX GPU and the numbers of DiffAgument are reproduced from the original code on our machine for a fair comparison.

| Model name                               | Dataset           | is10k     | fid10k    |
| -----------------------------------------| ------------------| --------- | --------- |
| BigGAN                                   | `C10` (10%)       | 7.03      | 48.3      |
| BigGAN-DiffAugment (translation + cutout)| `C10` (10%)       | 8.40      | 23.9      |
| BigGAN-DAG (rotation)                    | `C10` (10%)       | 7.87      | 36.9      |
| BigGAN-DAG (rotation + cropping)         | `C10` (10%)       | 8.63      | 23.6      |
| BigGAN-DAG (rotation + cropping + translation + cutout)         | `C10` (10%)       | 8.65      | 21.3      |
| BigGAN                                   | `C10` (20%)       | 8.51      | 22.3      |
| BigGAN-DiffAugment (translation + cutout)| `C10` (20%)       | 8.79      | 14.6      |
| BigGAN-DAG (rotation)                    | `C10` (20%)       | 8.98      | 17.6      |
| BigGAN-DAG (rotation + cropping)         | `C10` (20%)       | 8.83      | 14.1      |
| BigGAN-DAG (rotation + cropping + translation + cutout)         | `C10` (20%)       | 8.84      | 13.1      |

In the above results, we demonstrate that our model can be applied to multiple types of augmentations. Moreover, when combining more augmentations, e.g., our (rotation + cropping) and (translation + cutout) of DiffAugment, DAG can improve the baselines more significantly. To apply DAG for your GAN model, you may use above augmentations or simply design yourself augmentations suitable for your problems.

*To be updated soon with other augmentations and datasets.*


