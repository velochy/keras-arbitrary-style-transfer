from keras.models import Model,load_model,model_from_json
import keras.backend as K
import os, sys
from keras import regularizers, activations, losses
from model import *
from custom_layers import *
from keras.preprocessing.image import load_img, img_to_array, array_to_img

cdim, sdim, style_n = (256,256,3), (224,224,3), 100

if len(sys.argv) < 4:
    print("Usage: train <model.mdl> <content_img> <style_img>")

# Load models if they are already present
with open(sys.argv[1]+".json") as json_file:
    model = model_from_json(json_file.read(),custom_objects=CUSTOM_OBJECTS)
model.load_weights(sys.argv[1]+'.h5')

model.summary()

smodel = model.get_layer('model_1')
#smodel.summary()

simage = load_img(sys.argv[3], target_size=sdim)
simage = img_to_array(simage)/255.0
simage = simage.reshape((1, simage.shape[0], simage.shape[1], simage.shape[2]))
style = smodel.predict(simage)
print("Style %r" % (list(style[0])))

tmodel = model.get_layer('model_2')
#tmodel.summary()

cimage = load_img(sys.argv[2], target_size=cdim)
cimage = img_to_array(cimage)/255.0
cimage = cimage.reshape((1, cimage.shape[0], cimage.shape[1], cimage.shape[2]))
timg = tmodel.predict([cimage,style])

timg = timg.reshape(cdim)
timg = array_to_img(timg*255.0)
timg.show()

