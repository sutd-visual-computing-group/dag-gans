# Data Augmentation optimized for GAN (DAG) - Official implementation

> **On Data Augmentation for GAN Training** <br>
> Ngoc-Trung Tran, Viet-Hung Tran, Ngoc-Bao Nguyen, Trung-Kien Nguyen, Ngai-Man Cheung <br>
> [https://arxiv.org/abs/2006.05338](https://arxiv.org/abs/2006.05338)

> **Abstract:** *Recent successes in Generative Adversarial Networks (GAN) have affirmed the importance of using more data in GAN training. Yet it is expensive to collect data in many domains such as medical applications. Data Augmentation (DA) has been applied in these applications. In this work, we first argue that the classical DA approach could mislead the generator to learn the distribution of the augmented data, which could be different from that of the original data. We then propose a principled framework, termed Data Augmentation Optimized for GAN (DAG), to enable the use of augmented data in GAN training to improve the learning of the original distribution. We provide theoretical analysis to show that using our proposed DAG aligns with the original GAN in minimizing the Jensen–Shannon (JS) divergence between the original distribution and model distribution.  Importantly, the proposed DAG effectively  leverages the augmented data to improve the learning of discriminator and generator. We conduct experiments to apply DAG to different GAN models: unconditional GAN, conditional GAN, self-supervised GAN and CycleGAN using datasets of natural images and medical images. The results show that DAG achieves consistent and considerable improvements across these models. Furthermore, when DAG is used in some GAN models, the system establishes state-of-the-art Fr\'echet Inception Distance (FID) scores.*

## Introduction

- We provide simple implementations of the DAG modules in both PyTorch and TensorFlow, which can be easily integrated into any GAN models to improve the performance, especially in the case of limited data. We only illustrate some augmentation techniques (rotation, cropping, flipping, ...) as discussed in our paper, but our DAG is not limited to these augmentations. The more augmentation to be used, the better improvements DAG enhances the GAN models. It is also easy to design your augmentations within the modules. However, there may be a trade-off between the numbers of many augmentations to be used in DAG and the computational cost.

- It is also important to note that our model works well with any data augmentation techniques, either invertible (rotation, flipping) or non-invertible (translation, cropping, cutout, ...) and if the augmentation is invertible, the convergence is theoretically guaranteed.

- We also provide implementations to improve: i) [BigGAN backbone](https://github.com/sutd-visual-computing-group/dag-gans/tree/main/pytorch/examples/biggan) and compare one of our concurrent works, Data-efficient GAN [1], ii) image-image translation with [CycleGAN baseline](https://github.com/sutd-visual-computing-group/dag-gans/tree/main/pytorch/examples/cyclegan). 

## Citation

```
@article{tran2020dag,
  title={On Data Augmentation for GAN Training},
  author={Tran, Ngoc-Trung and Tran, Viet-Hung and Nguyen, Ngoc-Bao and Nguyen, Trung-Kien and Cheung, Ngai-Man},
  journal={arXiv preprint arXiv:2006.05338},
  year={2020}
}
```

## Reference

[1] Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han, "Differentiable Augmentation for Data-Efficient GAN Training", NeurIPS 2020.


*To be updated*





