# Data Augmentation optimized for GAN (DAG)

## How to use

Our tensorflow module is easily integrated into any GAN models with three following simple steps.

### tensorflow

#### Step 1: Including the DAG module into your code

To copy our *dag* module into your workspace and include the module into your code as follows:

```python
from dag.dag import DAG

... 

dag = DAG(D_loss_func, G_loss_func, policy=['rotation'], policy_weight=[1.0])
n_augments = dag.get_n_augments_from_policy()
...

loss_d += dag.compute_discriminator_loss(x_real, x_fake, netD)
loss_g += dag.compute_generator_loss(x_real, x_fake, netD)

```
- *policy*: the augmentation methods to be used in DAG.
- *policy_weight*: the corresponding weights for the augmentions to used in DAG.
- *D_loss_func*: the function of the discriminator loss (see step 3).
- *G_loss_func*: the function of the generator loss (see step 3).
- *get_n_augments_from_policy()*: to return the number of heads to implement DAG in the discriminator.
- *netD*: the discriminator network (see step 2)
- x_real: the batch of original real samples.
- x_fake: the batch of original fake samples.

#### Step 2: Modifying the outputs of the discriminator

To alternate the discriminator with multiple heads according to the number of augmentations returned from *get_n_augments_from_policy()*:

```python
class Discriminator(nn.Module):
    def __init__(self, n_augments):
        super(Discriminator, self).__init__()
        ...
        self.n_augments = n_augments
        ...
        self.linears_dag = []
        for i in range(self.n_augments):
            self.linears_dag.append(<Your linear head>)
        self.linears_dag = tf.concat(self.linears_dag, 0)
        ...
```

Note that the modified discriminator has different heads (but all other layes are shared), you need to modify the call of function in your code. For example:

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

To use DAG, we need the functions of computing losses of D and G to apply it automatically on augmented samples.


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
    d_cost = tf.reduce_mean(tf.nn.softplus(d_real)) + tf.reduce_mean(tf.nn.softplus(-d_fake))
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
    g_cost = tf.reduce_mean(tf.nn.softplus(-fake_scores))
    return g_cost
```

To use more augmentation techniques: 

```python
from dag.dag import DAG

... 

dag  = DAG(D_loss_func, G_loss_func, augment_type=['rotation', 'cropping'], augment_weight=[1.0, 1.0])
n_augments = dag.get_n_augments_from_policy()
...

```

