import torch

def rotation(x):
    return [x, x, x, x]
def fliprot(x):
    return [x, x, x, x]
def cropping(x):
    return [x, x, x, x]
def augmenting_data(x, aug_type='rotation'):
	if aug_type == 'rotation':
	   return rotation(x)
	elif aug_type == 'fliprot':
       return fliprot(x)
	elif aug_type == 'cropping':
	   return cropping(x)
	else:
	   print('The augmentation type is not supported. Exiting ...')
	   exit()