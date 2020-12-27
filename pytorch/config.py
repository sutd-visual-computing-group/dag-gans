''' rotation degree '''
rotations = [0, 90, 180, 270]
fliprot   = ['noflip', 'left-right', 'bottom-up', 'rotate90']
cropping  = ['nocrop', 'corner1', 'corner2', 'corner3', 'corner4']
augment_list = {
                 'rotation': rotations,
                 'fliprot' : fliprot,
                 'cropping' : cropping
               }
