from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import os, sys
from keras import regularizers, activations, losses, optimizers
from model import extract_features
from custom_layers import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from utils import load_json_h5_model
import numpy as np

# Has three outputs: 'result' (for total variation loss)
#  and 'ContentError' / 'StyleError' for respective errors
# Model assumes a dummy input of np.ones() the shape of style
def optimize_style_model(tmodel, cimage,simage):

    sdim, cdim = simage.shape, cimage.shape

    fcmodel = extract_features(cdim)
    fcmodel.trainable = False

    if cdim==sdim: fsmodel = fcmodel
    else:
        fsmodel = extract_features(sdim)
        fsmodel.trainable = False

    cimage = cimage.reshape((1, cimage.shape[0], cimage.shape[1], cimage.shape[2]))
    cvals = fcmodel.predict(cimage)[0][0]

    simage = simage.reshape((1, simage.shape[0], simage.shape[1], simage.shape[2]))
    svals = fsmodel.predict(simage)[1][0]

    cinp = Input(shape=cimage.shape[1:],dtype="float32",name="content_img")
    style_shape = tmodel.input_shape[1][1:]
    dummy = Input(shape=style_shape,dtype="float32",name="dummy_style")

    from keras import initializers
    style = RawWeights(name='style',activation=None,
                        initializer=initializers.uniform(-0.01,0.01))(dummy)

    timg = tmodel([cinp,style])
    res = Lambda(lambda x:x,name="result")(timg)

    tfeats = fcmodel(timg)

    cerr = Lambda(lambda x: x-cvals, name="ContentError")(tfeats[0])
    serr = Lambda(lambda x: x-svals, name="StyleError")(tfeats[1])

    model = Model(inputs=[cinp,dummy],outputs=[res,cerr,serr])
    return model

cdim, sdim = (256,256,3), (256,256,3)

# Begin actual functionality
if len(sys.argv) < 4:
    print("Usage: train <model> <content_img> <style_img>")

# Load models if they are already present
lmodel = load_json_h5_model(sys.argv[1])
tmodel = lmodel.get_layer('model_transfer')
tmodel.trainable = False

styledim = tmodel.input_shape[1][1:]

simage = load_img(sys.argv[3],target_size=sdim)
simage = img_to_array(simage)
#simage.show()

cimage = load_img(sys.argv[2],target_size=cdim)
cimage = img_to_array(cimage)
cdim = cimage.shape
#cimage.show()

print("Building model")
model = optimize_style_model(tmodel,cimage,simage)

model.compile(optimizer=optimizers.adam(learning_rate=1e-2), 
              loss={ 'ContentError': mean_squared_value, 'StyleError': mean_squared_value, 'result':total_variation_loss },
              loss_weights={'ContentError': 3.0, 'StyleError': 10.0, 'result':1.0})

print("Model built")
model.summary()

# Initialize result with content image instead of random noise
#model.get_layer('result').set_weights([cimage])

n_iterations = 100
done_epochs = 0
for i in range(n_iterations):
	N = 300
	model.fit([np.repeat(cimage[np.newaxis,:,:,:],N,axis=0), np.ones((N,)+styledim)],[np.zeros(N)]*3,batch_size=1,
		epochs=done_epochs+1, initial_epoch=done_epochs)

	# Display result
	timg = model.predict([cimage[np.newaxis,:,:,:], np.ones((1,)+styledim)])[0]
	timg = timg.reshape(cdim)
	timg = np.minimum(255,np.maximum(0,np.round(timg)))
	timg = array_to_img(timg)
	timg.show()
	done_epochs += 1
