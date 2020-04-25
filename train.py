from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import os, sys
from keras import regularizers, activations, losses, optimizers
from model import *
from utils import DataGenerator, load_json_h5_model, save_json_h5_model
from custom_layers import *

cdim, sdim, style_n = (256,256,3), (256,256,3), 100

if len(sys.argv) < 2:
    print("Usage: train <model>")

# Load models if they are already present
if os.path.exists(sys.argv[1]+'.json'):
    model = load_json_h5_model(sys.argv[1])

    # Submodel level trainability not preserved in save :(
    smodel = model.get_layer('model_style')
    tmodel = model.get_layer('model_transfer')
    
    # Rebuild the learning metric part
    model = full_model(tmodel,smodel)

    # Also leave the mobilenet as-is for now 
    #smodel = model.get_layer('model_1')
    #for l in smodel.layers[:-1]:
    #    l.trainable = False
else:
    smodel = style_model(sdim,style_n)
    #smodel.summary()
    tmodel = transfer_model(cdim,style_n)
    #tmodel.summary()
    model = full_model(tmodel,smodel)

model.summary()

model.compile(optimizer=optimizers.adam(),
              loss={ 'ContentError': mean_squared_value, 'StyleError': mean_squared_value, 'result':total_variation_loss },
              loss_weights={'ContentError': 3.0, 'StyleError': 10.0, 'result':1.0})


#cdir, sdir = './toy_content/','./toy_styles/'
#cdir, sdir = './content_images/','./toy_styles/'
cdir, sdir = './content_images/','./style_images/'
BATCH_SIZE = 4
BATCH_EPOCHS = 1

total_epochs = 5000 
done_epochs = 0
while total_epochs>0:
    epochs = min(total_epochs, BATCH_EPOCHS)

    print("Saving model")
    save_json_h5_model(sys.argv[1])

    print("Training model")
    model.fit_generator(DataGenerator(cdir,sdir,cdim,sdim,BATCH_SIZE),
            #use_multiprocessing=True, workers=4,
            epochs=epochs+done_epochs, initial_epoch=done_epochs)
    

    total_epochs -= epochs
    done_epochs += epochs