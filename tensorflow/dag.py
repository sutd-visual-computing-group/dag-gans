from .utils  import *
from .config import *

import tensorflow as tf

def get_n_augments_from_policy(policy):
    n_augments_per_policy = 0
    for i in range(len(policy)):
        n_augments_per_policy += len(augment_list[policy[i]])
    return n_augments_per_policy

class DAG(object):
    def __init__(self, D_loss_func, G_loss_func, policy=['rotation'], policy_weight=[1.0]):
        self.D_loss_func     = D_loss_func
        self.G_loss_func     = G_loss_func
        self.policy          = policy
        self.policy_weight   = policy_weight

    def get_n_augments_from_policy(self):
        n_augments_per_policy = 0
        for i in range(len(self.policy)):
            n_augments_per_policy += len(augment_list[self.policy[i]])
        return n_augments_per_policy

    def get_augmented_samples(self, x):
        x_arg = []
        n_policies = len(self.policy)
        for aug_type in self.policy:
            x_arg.append(augmenting_data(x, aug_type, augment_list[aug_type]))
        x_arg = tf.concat(x_arg,0)
        return x_arg

    def compute_discriminator_loss(self, x_real, x_fake, netD):
          
        ''' compute D loss for original augmented real/fake data samples '''
        d_loss = 0
        n_policies = len(self.policy)
        policy_index  = 0
        weight = 0
        for i in range(len(self.policy)):
            x_real_aug = augmenting_data(x_real, self.policy[i], augment_list[self.policy[i]])
            x_fake_aug = augmenting_data(x_fake, self.policy[i], augment_list[self.policy[i]])
            weight = self.policy_weight[i]
            n_policy = len(augment_list[self.policy[i]])
            d_loss_per_policy = 0
            for j in range(n_policy):
                d_loss_per_policy += self.D_loss_func(x_real_aug[j], x_fake_aug[j], netD, dag=True, dag_idx=policy_index+j)
            d_loss = d_loss + weight * d_loss_per_policy / n_policy
            policy_index += n_policy
        return d_loss

    def compute_generator_loss(self, x_real, x_fake, netD):
          
        ''' compute G loss for original augmented real/fake data samples '''
        g_loss = 0
        n_policies = len(self.policy)
        policy_index  = 0
        weight    = 0        
        for i in range(len(self.policy)):
            x_real_aug = augmenting_data(x_real, self.policy[i], augment_list[self.policy[i]])
            x_fake_aug = augmenting_data(x_fake, self.policy[i], augment_list[self.policy[i]])
            weight = self.policy_weight[i]
            n_policy = len(augment_list[self.policy[i]])
            g_loss_per_policy = 0
            for j in range(n_policy):
                g_loss_per_policy += self.G_loss_func(x_real_aug[j], x_fake_aug[j], netD, dag=True, dag_idx=policy_index+j)
            g_loss = g_loss + weight * g_loss_per_policy / n_policy
            policy_index += n_policy
        return g_loss

    def compute_discriminator_logits(self, inputs, netD):
          
        ''' computing the discriminator logits over the inputs '''
        logits   = []
        augments = []
        n_policies = len(self.policy)
        policy_index  = 0
        weight    = 0
        for i in range(len(self.policy)):
            inputs_aug = augmenting_data(inputs, self.policy[i], augment_list[self.policy[i]])
            n_augments_per_policy = len(augment_list[self.policy[i]])
            for j in range(n_augments_per_policy):
                _, logits_tmp = netD(inputs_aug[j])
                augments.append(inputs_aug[j])
                logits.append(logits_tmp[j])
                #real_grads = tf.gradients(tf.reduce_sum(logits[-1]), [augments[-1]])[0]
                #gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
        #augments = tf.stack(augments,0)
        #logits = tf.stack(logits,0)

        return logits, augments

    def compute_loss_from_logits(self, real_logits, fake_logits, loss_func):
          
        ''' compute D loss for original augmented real/fake data samples '''
        loss    = 0
        n_policies  = len(self.policy)
        policy_index = 0
        weight   = 0
        for i in range(len(self.policy)):
            weight = self.policy_weight[i]
            n_augments_per_policy = len(augment_list[self.policy[i]])
            d_loss_per_policy = 0
            for j in range(n_augments_per_policy):
                d_loss_per_policy += loss_func(real_logits[policy_index+j], fake_logits[policy_index+j])
            loss = loss + weight * d_loss_per_policy / n_augments_per_policy
            policy_index += n_augments_per_policy
        return loss

    def compute_loss_from_logits_and_samples(self, reals, real_logits, fakes, fake_logits, loss_func):
          
        ''' compute D loss for original augmented real/fake data samples '''
        loss    = 0
        n_policies   = len(self.policy)
        policy_index = 0
        weight  = 0
        for i in range(len(self.policy)):
            weight = self.policy_weight[i]
            n_augments_per_policy = len(augment_list[self.policy[i]])
            loss_per_policy = 0
            for j in range(n_augments_per_policy):
                loss_per_policy += loss_func(reals[policy_index+j], real_logits[policy_index+j], fakes[policy_index+j], fake_logits[policy_index+j])
            loss = loss + weight * loss_per_policy / n_augments_per_policy
            policy_index += n_augments_per_policy
        return loss

'''
if __name__ == "__main__":
    
    import argparse
    import tensorflow as tf
    from cifar import load_cifar10
    from skimage import io
    import numpy as np
 
    #command lines
    parser = argparse.ArgumentParser(description='Tensorflow code test transformations with the cifar datasets.')
    parser.add_argument('--dataset', default='cifar10',  type=str, help='The dataset name (cifar10 or cifar100).')
    parser.add_argument('--source',  default='data/cifar-10-batches-py/',  type=str, help='The source of cifar datasets.')
    parser.add_argument('--augment', default='cropping',  type=str, help='The augmentation method.')
    args = parser.parse_args()
    
    #load CIFAR dataset
    source       = args.source
    if args.dataset == 'cifar10':
       data_files   = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
       images, labels = load_cifar10(source, data_files)
    elif args.dataset == 'cifar100':
       data_files   = ['train']
       images, labels = load_cifar100(source, data_files)

    #declare DAG
    X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    policy=[args.augment]
    dag   = DAG(None, None, policy=policy)
    X_aug = dag.get_augmented_samples(X)

    with tf.Session() as sess:
        #test the transformations
        for i in range(len(images)):
            io.imsave('output_{}_{}_real.png'.format(policy[0], i), images[i])
            inputs_aug = sess.run(X_aug, feed_dict={X: np.reshape(images[i],(-1,32,32,3))})
            for j in range(len(inputs_aug)):
                io.imsave('output_{}_{}_transformed.png'.format(policy[0], j), np.reshape(inputs_aug[j],(32,32,3)))
            break
'''
