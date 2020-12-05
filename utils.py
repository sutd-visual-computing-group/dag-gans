import torch

def rotation(x, degs):
    x_rot = []
    for deg in degs:
        if deg == 0:
           x_rot.append(x)
        elif deg == 90:
           x_rot.append(x.flip(2))
        elif deg == 180:
           x_rot.append(x.flip(2).flip(3))
        elif deg == 270:
           x_rot.append(x.flip(3))
    return x_rot
    
def fliprot(x, aug):
    return [x, x, x, x]

def cropping(x, aug):
    return [x, x, x, x]

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