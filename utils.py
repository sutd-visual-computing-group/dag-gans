import torch
import torch.nn.functional as F
import numpy as np

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
    #x_rot = torch.cat(x_rot,0)
    return x_rot
    
def fliprot(x, aug):
    x_flip = []
    x_flip.append(x)
    x_flip.append(x.flip(2))
    x_flip.append(x.flip(3))
    x_flip.append(x.transpose(2, 3).flip(2))
    #x_flip = torch.cat(x_flip,0)
    return x_flip

def cropping(x, aug):
    b, c, h, w = x.shape
    boxes = [[0,      0,      h,      w],
             [0,      0,      h*0.75, w*0.75],
             [0,      w*0.25, h*0.75, w],
             [h*0.25, 0,      h,      w*0.75],
             [h*0.25, w*0.25, h,      w]]
    x_crop = []
    for i in range(np.shape(boxes)[0]):
        cropped = x[:,:,int(boxes[i][0]):int(boxes[i][2]), int(boxes[i][1]):int(boxes[i][3])].clone()
        x_crop.append(F.interpolate(cropped, (h, w)))
    #x_crop = torch.cat(x_crop,0)
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