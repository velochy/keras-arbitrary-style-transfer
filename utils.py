from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.models import model_from_json
from custom_layers import CUSTOM_OBJECTS
import numpy as np

class DataGenerator(Sequence):

    def __init__(self, cdir, sdir, cdim, sdim, batch_size):
        cidg = ImageDataGenerator(#rescale = 1./255, 
                                   #shear_range = 0.2, 
                                   zoom_range = [0.5,1],
                                   rotation_range=60.,
                                   horizontal_flip = True)

        self.cgen = cidg.flow_from_directory(cdir,
                                              target_size = (cdim[0],cdim[1]),
                                              class_mode = 'categorical',
                                              batch_size = batch_size)

        sidg = ImageDataGenerator(#rescale = 1./255,
                                   shear_range = 0.2, 
                                   zoom_range = [0.5,1],
                                   rotation_range=60.,
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
        return [cimg[0], simg[0]], [self.y,self.y,self.y]  #Yield both images and three dummy len-1 arrays as y-s


def change_input_shapes(model, new_input_shapes):

    # replace input shape of first layer
    for i,layer in enumerate(filter(lambda l: isinstance(l,InputLayer), model._layers)):
      layer._batch_input_shape = layer.batch_input_shape = new_input_shapes[i]
      print("LAYER",layer,layer.batch_input_shape,layer.input_shape)

    # rebuild model architecture by exporting and importing via json
    new_model = model_from_json(model.to_json(),custom_objects=CUSTOM_OBJECTS)
    #new_model.summary()

    # copy weights from old model to new one - can take a bit of time
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model

def save_json_h5_model(model, fname):
    model.save_weights(fname+'.h5')
    model_json = model.to_json()
    with open(fname+".json", "w") as json_file:
        json_file.write(model_json)

def load_json_h5_model(fname):
    with open(fname+".json") as json_file:
      model = model_from_json(json_file.read(),custom_objects=CUSTOM_OBJECTS)
    model.load_weights(fname+'.h5')
    return model

