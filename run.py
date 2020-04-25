from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import os, sys
from keras import regularizers, activations, losses
from model import *
from custom_layers import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from utils import change_input_shapes, load_json_h5_model

# Force CPU as GPU loading takes a lot of time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

cdim, sdim = (256,256,3), (256,256,3)

if len(sys.argv) < 4:
    print("Usage: run <model> <content_img> <style_img>")

print("Loading model")
# Load models if they are already present
model = load_json_h5_model(sys.argv[1])
#model.summary()

smodel = model.get_layer('model_style')
tmodel = model.get_layer('model_transfer')

#print("Reshaping models")
#smodel = change_input_shapes(smodel, [(None,None,None,3)])
#tmodel = change_input_shapes(tmodel, [(None,None,None,3),tmodel.input_shape[1]])

#smodel.summary()
#tmodel.summary()

sdim, cdim = smodel.input_shape[1:], tmodel.input_shape[0][1:] 

simage = load_img(sys.argv[3], target_size=sdim)
#simage.show()
simage = img_to_array(simage)
simage = simage.reshape((1, simage.shape[0], simage.shape[1], simage.shape[2]))

print("Calculating style")
style = smodel.predict(simage)
print("Style %r" % (list(style[0])))


cimage = load_img(sys.argv[2], target_size=cdim)
#cimage.show()
cimage = img_to_array(cimage)
cimage = cimage.reshape((1, cimage.shape[0], cimage.shape[1], cimage.shape[2]))

print("Styling content image")
timg = tmodel.predict([cimage,style])

timg = timg.reshape(timg.shape[1:])
timg = array_to_img(timg)
timg.show()

