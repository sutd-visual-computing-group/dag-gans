import torch

from utils import augmenting_data

rotations = [0, 90, 180, 270]

augment_list = {
                 'rotation': rotations
               } 

class DAG(object):
    def __init__(self, D_loss_func, G_loss_func, augument_type=['rotation']):
        print('Initializing DAG ...')
        self.D_loss_func   = D_loss_func
        self.G_loss_func   = G_loss_func
        self.augument_type = augument_type

    def get_num_of_augments():
        num_of_augments = 0
        for i in range(len(self.augument_type)):
            num_of_augments += len(augment_list[self.augument_type[i]])
        return num_of_augments

    def dag_loss(self, x_real, x_fake):
          
        ''' compute D loss and G loss for original augmented real/fake data samples '''
        d_loss = 0
        g_loss = 0
        
        n_type = len(self.augument_type)
        
        for aug_type in self.augument_type:
            x_real_aug = augmenting_data(x_real, aug_type)
            x_fake_aug = augmenting_data(x_fake, aug_type)
            n_aug_type = len(x_real_aug)
            for i in range(n_aug_type):
                d_loss += self.D_loss_func(x_real_aug[i], x_fake_aug[i])
                g_loss += self.G_loss_func(x_real_aug[i], x_fake_aug[i])
        
        d_loss = d_loss / n_type
        g_loss = g_loss / n_type

        return d_loss, g_loss
          