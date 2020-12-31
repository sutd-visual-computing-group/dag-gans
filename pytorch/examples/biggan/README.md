We implement our DAG on its BigGAN based on [DiffAugment](https://github.com/mit-han-lab/data-efficient-gans). 

We share our BigGAN + DAG code. Simply substituting these files in the original code. We also provide the script examples to run our DAG. Here, we highlight our performance as comparing to this work. Note that our model is trained on Single Titan RTX GPU and these numbers are reproduced from the original code for fair comparision.

| Model name                               | Dataset           | is10k     | fid10k    |
| -----------------------------------------| ------------------| --------- | --------- |
| BigGAN                                   | `C10` (10%)       | 7.03      | 48.3      |
| BigGAN-DiffAugment (translation + cutout)| `C10` (10%)       | 8.40      | 23.9      |
| BigGAN-DAG (rotation + cropping)         | `C10` (10%)       | 8.63      | 23.6      |
| BigGAN                                   | `C10` (20%)       | 8.51      | 22.3      |
| BigGAN-DiffAugment (translation + cutout)| `C10` (20%)       | 8.79      | 14.6      |
| BigGAN-DAG (rotation + cropping)         | `C10` (20%)       | 8.83      | 14.1      |

*To be updated soon with other data augmentations and other datasets.*


