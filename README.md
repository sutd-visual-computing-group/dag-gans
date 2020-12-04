# Data Augmentation optimized for GAN (DAG)

We provide the DAG modules in pytorch and tensorflow, which can be easily integrated into any GAN models to improve the performance, especially in the case of limited data.

## How to use

### Pytorch

#### Step 1: Including the DAG module into your code

To copy our *dag* module into your workspace and include the module into your code as follows:

```python
from dag.dag_pytorch import DAG

... 

dag = DAG(D_loss_func, G_loss_func, augment_type=['rotation', 'cropping'])
n_augments = dag.get_num_of_augments()
...

```
- *augument_type*: the augmentation methods to be used in DAG.
- *D_loss_func*: the function of the discriminator loss (see step 2).
- *G_loss_func*: the function of the generator loss (see step 2).
- *get_num_of_augments()*: to return the number of heads to implement DAG in the discriminator.

#### Step 2: Defining the loss functions for discriminator and generator

To use DAG, we need the functions of computing losses of D and G to apply it automatically on augmented samples:

```python
def D_loss_func(x_real, x_fake):
    return D
```

```python
def G_loss_func(x_real, x_fake):
    return G
```

#### Step 3: Modifying the outputs of the discriminator

To modify the discriminator architecture according to the number of augmentations used in the DAG. For example:

The original discriminator:

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ...
        self.linear = nn.Linear(4*4*4*DIM, 1)
        ...

    def forward(self, input):
        ...
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output
```

The modified discriminator:

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self, n_augments).__init__()
        ...
        self.n_augments = n_augments
        ...
        self.linear = nn.Linear(4*4*4*DIM, 1)
        self.linears_dag = []
        for i in range(self.n_augments):
            self.linears_dag[i].append(nn.Linear(4*4*4*DIM, 1))

    def forward(self, input):
        ...
        feature = output.view(-1, 4*4*4*DIM)
        output  = self.linear(feature)
        # dag heads
        outputs_dag = []
        for i in range(self.n_augments):
            outputs_dag[i].append(self.linears_dag[i](feature))
        return output, outputs_dag
```


