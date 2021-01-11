import torch

from   dag.utils import augmenting_data
import dag.config as config

class DAG(object):
    def __init__(self, D_loss_func, G_loss_func, augument_type=['rotation'], augument_weight=[1.0]):
        print('Initializing DAG ...')
        self.D_loss_func   = D_loss_func
        self.G_loss_func   = G_loss_func
        self.augument_type = augument_type
        self.augument_weight = augument_weight

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
        x_arg = torch.cat(x_arg,0)
        return x_arg

    def compute_discriminator_loss(self, x_real, x_fake, netD):
          
        ''' compute D loss for original augmented real/fake data samples '''
        d_loss = 0
        n_type = len(self.augument_type)
        aug_idx  = 0
        aug_w    = 0
        for i in range(len(self.augument_type)):
            x_real_aug = augmenting_data(x_real, self.augument_type[i], config.augment_list[self.augument_type[i]])
            x_fake_aug = augmenting_data(x_fake, self.augument_type[i], config.augment_list[self.augument_type[i]])
            aug_w = self.augument_weight[i]
            n_aug_type = len(config.augment_list[self.augument_type[i]])
            d_loss_aug = 0
            for j in range(n_aug_type):
                d_loss_aug += self.D_loss_func(x_real_aug[j], x_fake_aug[j], netD, dag=True, dag_idx=aug_idx+j)
            d_loss = d_loss + aug_w * d_loss_aug / n_aug_type
            aug_idx += n_aug_type
        return d_loss

    def compute_generator_loss(self, x_real, x_fake, netD):
          
        ''' compute G loss for original augmented real/fake data samples '''
        g_loss = 0
        n_type = len(self.augument_type)
        aug_idx  = 0
        aug_w    = 0        
        for i in range(len(self.augument_type)):
            x_real_aug = augmenting_data(x_real, self.augument_type[i], config.augment_list[self.augument_type[i]])
            x_fake_aug = augmenting_data(x_fake, self.augument_type[i], config.augment_list[self.augument_type[i]])
            aug_w = self.augument_weight[i]
            n_aug_type = len(config.augment_list[self.augument_type[i]])
            g_loss_aug = 0
            for j in range(n_aug_type):
                g_loss_aug += self.G_loss_func(x_real_aug[j], x_fake_aug[j], netD, dag=True, dag_idx=aug_idx+j)
            g_loss = g_loss + aug_w * g_loss_aug / n_aug_type
            aug_idx += n_aug_type
        return g_loss

if __name__ == "__main__":

    import torchvision
    import torchvision.utils as vutils
    import torchvision.transforms as transforms
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    testset    = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=6, drop_last=True)

    augument_type=['cropping']
    dag = DAG(None, None, augument_type=augument_type)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_aug = dag.get_augmented_samples(inputs)
        for i in range(len(inputs_aug)):
            vutils.save_image(torch.tensor(inputs_aug[i]), 'output_{}_{}.png'.format(augument_type[0], i), normalize=True, scale_each=True, nrow=10)
        break
        
