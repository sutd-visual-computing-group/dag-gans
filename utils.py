import torch

def rotation(x, degs):
    x_rot = []
    for deg in degs:
        if deg == 0:
           x_rot.append(x)
        elif deg == 90:
           x_rot.append(x.transpose(2, 3).flip(2))
        elif deg == 180:
           x_rot.append(x.flip(2).flip(3))
        elif deg == 270:
           x_rot.append(x.transpose(2, 3).flip(3))
    return x_rot
    
def fliprot(x, aug):
    x_flip = []
    x_flip.append(x)
    x_flip.append(x.flip(2))
    x_flip.append(x.flip(3))
    x_flip.append(x.transpose(2, 3).flip(2))
    return x_flip

def cropping(x, aug):
    x_crop = []
    x_crop.append(x)
    return x_crop

def augmenting_data(x, aug, aug_list):
    if aug == 'rotation':
       return rotation(x, aug_list)
    elif aug == 'fliprot':
       return fliprot(x, aug_list)
    elif aug == 'cropping':
       return cropping(x, aug_list)
    else:
       print('utils.augmenting_data: the augmentation type is not supported. Exiting ...')
       exit()