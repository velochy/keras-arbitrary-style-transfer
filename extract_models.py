from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import os, sys
from keras import regularizers, activations, losses
import tensorflow as tf
from model import *
from custom_layers import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from utils import change_input_shapes, load_json_h5_model, save_json_h5_model

# Force CPU as GPU loading takes a lot of time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

cdim, sdim = (256,256,3), (256,256,3)

if len(sys.argv) < 4:
    print("Usage: extract_models <combined model> <style model> <transfer model>")

print("Loading full model")
model = load_json_h5_model(sys.argv[1])

#model.summary()
smodel = model.get_layer('model_style')
tmodel = model.get_layer('model_transfer')

print("Reshaping model inputs")
smodel = change_input_shapes(smodel, [(None,None,None,3)])
smodel.summary()

tmodel = change_input_shapes(tmodel, [(None,None,None,3),tmodel.input_shape[1]])
tmodel.summary()

print("Saving models separately")
save_json_h5_model(smodel,sys.argv[2])
save_json_h5_model(tmodel,sys.argv[3])

#smodel.save(sys.argv[2])
#tmodel.save(sys.argv[2])