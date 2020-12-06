import torch

from   dag.utils import augmenting_data
import dag.config as config


class DAG(object):
    def __init__(self, D_loss_func, G_loss_func, augument_type=['rotation']):
        print('Initializing DAG ...')
        self.D_loss_func   = D_loss_func
        self.G_loss_func   = G_loss_func
        self.augument_type = augument_type

    def get_num_of_augments(self):
        num_of_augments = 0
        for i in range(len(self.augument_type)):
            num_of_augments += len(config.augment_list[self.augument_type[i]])
        return num_of_augments

    def get_augmented_samples(self, x):
        x_arg = []
        n_type = len(self.augument_type)
        for aug_type in self.augument_type:
            x_arg.append(augmenting_data(x, aug_type, config.augment_list[aug_type]))
        return x_arg

    def compute_discriminator_loss(self, x_real, x_fake, netD):
          
        ''' compute D loss for original augmented real/fake data samples '''
        d_loss = 0
                
        n_type = len(self.augument_type)
        
        for aug_type in self.augument_type:
            x_real_aug = augmenting_data(x_real, aug_type, config.augment_list[aug_type])
            x_fake_aug = augmenting_data(x_fake, aug_type, config.augment_list[aug_type])
            n_aug_type = len(x_real_aug)
            for i in range(n_aug_type):
                d_loss += self.D_loss_func(x_real_aug[i], x_fake_aug[i], netD)
        
        d_loss = d_loss / n_type
        
        return d_loss

    def compute_generator_loss(self, x_real, x_fake, netD):
          
        ''' compute G loss for original augmented real/fake data samples '''
        g_loss = 0
        
        n_type = len(self.augument_type)
        
        for aug_type in self.augument_type:
            x_real_aug = augmenting_data(x_real, aug_type, config.augment_list[aug_type])
            x_fake_aug = augmenting_data(x_fake, aug_type, config.augment_list[aug_type])
            n_aug_type = len(x_real_aug)
            for i in range(n_aug_type):
                g_loss += self.G_loss_func(x_real_aug[i], x_fake_aug[i], netD)
        
        g_loss = g_loss / n_type

        return g_loss


if __name__="__main__":

    dag = DAG(None, None, augument_type=['rotation'])
    
    dag.get_augmented_samples()