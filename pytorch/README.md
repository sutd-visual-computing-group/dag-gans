# Data Augmentation optimized for GAN (DAG)

## How to use

Our pytorch module is easily integrated into any GAN models with three following simple steps. We use this [wgan-gp pytorch code](https://github.com/caogang/wgan-gp) for illustration.

### Pytorch

#### Step 1: Including the DAG module into your code

To copy our *dag* module into your workspace and include the module into your code as follows:

```python
from dag.dag import DAG

... 

dag = DAG(D_loss_func, G_loss_func, policy=['rotation'], policy_weight=[1.0])
n_augments = dag.get_num_of_augments_from_policy()
...

loss_d += dag.compute_discriminator_loss(x_real, x_fake, netD)
loss_g += dag.compute_generator_loss(x_real, x_fake, netD)

```
- *augument_type*: the augmentation methods to be used in DAG.
- *policy_weight*: the corresponding weights for the augmentions to used in DAG.
- *D_loss_func*: the function of the discriminator loss (see step 3).
- *G_loss_func*: the function of the generator loss (see step 3).
- *get_num_of_augments_from_policy()*: to return the number of heads to implement DAG in the discriminator.
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
def D_loss_func(x_real, x_fake, netD, dag=False, dag_idx=0):
    if dag==False:
       d_real, _ = netD(x_real)
       d_fake, _ = netD(x_fake)
    else:
       _, d_reals = netD(x_real)
       d_real = d_reals[dag_idx]
       _, d_fakes = netD(x_fake)
       d_fake = d_fakes[dag_idx]
    d_real    = d_real.mean()
    d_fake    = d_fake.mean()
    # train with gradient penalty
    gp = calc_gradient_penalty(netD, x_real, x_fake, dag=dag, dag_idx=dag_idx)
    d_cost = d_fake - d_real + gp
    return d_cost
```

```python
def G_loss_func(x_real, x_fake, netD, dag=False, dag_idx=0):
    if dag==False:
       d_fake, _ = netD(x_fake)
    else:
       _, d_fakes = netD(x_fake)
       d_fake = d_fakes[dag_idx]
    d_fake    = d_fake.mean()
    g_cost = -d_fake
    return g_cost
```

- *dag*: the flag get the discriminator output of original or dag.
- *dag_idx*: the index of dag output to be used when *dag=True*

To use more augmentation techniques: 

```python
from dag.dag import DAG

... 

dag  = DAG(D_loss_func, G_loss_func, policy=['rotation', 'cropping'], policy_weight=[1.0, 1.0])
n_augments = dag.get_num_of_augments_from_policy()
...

```

