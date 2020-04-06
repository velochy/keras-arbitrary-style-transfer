from keras.models import Model,load_model,model_from_json
import keras.backend as K
import os, sys
from keras import regularizers, activations, losses
from model import *
from datagen import DataGenerator
from custom_layers import *

cdim, sdim, style_n = (256,256,3), (224,224,3), 100

if len(sys.argv) < 2:
    print("Usage: train <model.mdl>")

# Load models if they are already present
if os.path.exists(sys.argv[1]+'.json'):
    with open(sys.argv[1]+".json") as json_file:
        model = model_from_json(json_file.read(),custom_objects=CUSTOM_OBJECTS)
    model.load_weights(sys.argv[1]+'.h5')

    # Submodel level trainability not preserved in save :(
    model.get_layer('model_3').trainable = False
    model.get_layer('model_4').trainable = False
    model.summary()
else:
    model = full_model(cdim,sdim,style_n)

model.compile(optimizer='adam',
              loss={ 'ContentError': mean_squared_value, 'StyleError': mean_squared_value },
              loss_weights={'ContentError':1.0, 'StyleError': 3.0})


#cdir, sdir = './toy_content/','./toy_styles/'
#cdir, sdir = './content_images/','./toy_styles/'
cdir, sdir = './content_images/','./style_images/'
BATCH_SIZE = 6
BATCH_EPOCHS = 1

total_epochs = 5000 
done_epochs = 0
while total_epochs>0:
    epochs = min(total_epochs, BATCH_EPOCHS)

    print("Saving model")
    model.save_weights(sys.argv[1]+'.h5')
    model_json = model.to_json()
    with open(sys.argv[1]+".json", "w") as json_file:
        json_file.write(model_json)

    print("Training model")
    model.fit_generator(DataGenerator(cdir,sdir,cdim,sdim,BATCH_SIZE),
            epochs=epochs+done_epochs, initial_epoch=done_epochs)
    
    total_epochs -= epochs
    done_epochs += epochs