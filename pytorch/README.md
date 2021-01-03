# Data Augmentation optimized for GAN (DAG)

## How to use

Our pytorch module is easily integrated into any GAN models with three following simple steps. We use this [wgan-gp pytorch code](https://github.com/caogang/wgan-gp) for illustration.

### Pytorch

#### Step 1: Including the DAG module into your code

To copy our *dag* module into your workspace and include the module into your code as follows:

```python
from dag.dag import DAG

... 

dag = DAG(D_loss_func, G_loss_func, augment_type=['fliprot'])
n_augments = dag.get_num_of_augments()
...

loss_d += dag.compute_discriminator_loss(x_real, x_fake, netD)
loss_g += dag.compute_generator_loss(x_real, x_fake, netD)

```
- *augument_type*: the augmentation methods to be used in DAG.
- *D_loss_func*: the function of the discriminator loss (see step 3).
- *G_loss_func*: the function of the generator loss (see step 3).
- *get_num_of_augments()*: to return the number of heads to implement DAG in the discriminator.
- *netD*: the discriminator network (see step 2)
- x_real: the batch of original real samples.
- x_fake: the batch of original fake samples.

#### Step 2: Modifying the outputs of the discriminator

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
    def __init__(self, n_augments):
        super(Discriminator, self).__init__()
        ...
        self.n_augments = n_augments
        ...
        self.linear = nn.Linear(4*4*4*DIM, 1)
        self.linears_dag = []
        for i in range(self.n_augments):
            self.linears_dag.append(nn.Linear(4*4*4*DIM, 1))
        self.linears_dag = nn.ModuleList(self.linears_dag)

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

Note that the modified discriminator has mulitple outputs, you need to modify the number of outputs when calling discriminator function. For example:

From: 
```python
netD = Discriminator()
d_real = netD(x_real)
```
To:
```python
netD = Discriminator(n_augments=n_augments)
d_real, _ = netD(x_real)
```

#### Step 3: Defining the loss functions for discriminator and generator

To use DAG, we need the functions of computing losses of D and G to apply it automatically on augmented samples. Followings are examples of GAN and WGAN losses:

##### WGAN loss

```python
def D_loss_func(x_real, x_fake, netD):
    # real
    d_real, _ = netD(x_real)
    d_real    = d_real.mean()
    # fake    
    d_fake, _ = netD(x_fake)
    d_fake,   = d_fake.mean()
    # train with gradient penalty
    gp = calc_gradient_penalty(netD, x_real, x_fake)    
    # D cost
    d_cost = d_fake - d_real + gp
    return d_cost
```

```python
def G_loss_func(x_real, x_fake, netD):
    # fake    
    d_fake, _ = netD(x_fake)
    d_fake    = d_fake.mean()
    # D cost
    g_cost = -d_fake
    return g_cost
```

To use more augmentation techniques: 

```python
from dag.dag_pytorch import DAG

... 

dag_fliprot  = DAG(D_loss_func, G_loss_func, augment_type=['fliprot'])
dag_cropping = DAG(D_loss_func, G_loss_func, augment_type=['cropping'])
n_augments = dag_fliprot.get_num_of_augments() + dag_cropping.get_num_of_augments()
...

```

