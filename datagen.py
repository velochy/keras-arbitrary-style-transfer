from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):

    def __init__(self, cdir, sdir, cdim, sdim, batch_size):
        cidg = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.5,
                                   rotation_range=5.,
                                   horizontal_flip = True)

        self.cgen = cidg.flow_from_directory(cdir,
                                              target_size = (cdim[0],cdim[1]),
                                              class_mode = 'categorical',
                                              batch_size = batch_size)

        sidg = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2, 
                                   zoom_range = 0.5,
                                   rotation_range=5.,
                                   horizontal_flip = True)
        self.sgen = sidg.flow_from_directory(sdir,
                                              target_size = (sdim[0],sdim[1]),
                                              class_mode = 'categorical',
                                              batch_size = batch_size)
        
        # Using shuffle on flow_from_directory causes issues on last batch size
        # Get around this by (a) not using last batch of each and (b) randomizing ourselves
        self.cinds = np.random.permutation(len(self.cgen)-1)
        self.sinds = np.random.permutation(len(self.sgen)-1)

        self.y = np.zeros((batch_size,1))

    def __len__(self):
        return min(len(self.cgen),len(self.sgen))-1 # as last one would not have matching batch sizes

    def __getitem__(self, idx):
        cimg, simg = self.cgen[self.cinds[idx]], self.sgen[self.sinds[idx]]
        return [cimg[0], simg[0]], [self.y,self.y]  #Yield both images and two len-1 arrays as y-s