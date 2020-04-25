from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import os, sys
from keras import regularizers, activations, losses, optimizers
from model import extract_features
from custom_layers import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# Create a model based on Gatys2015
# Has three outputs: 'result' (for total variation loss)
#  and 'ContentError' / 'StyleError' for respective errors
# Model assumes a dummy input of np.ones() the shape of content image
def single_image_transform_model(cimage,simage):

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

    dummy = Input(shape=cdim,dtype="float32",name="dummy_img")
    from keras import initializers
    timg = RawWeights(name='result',activation=None,
                        initializer=initializers.uniform(0,255))(dummy)
    tfeats = fcmodel(timg)

    cerr = Lambda(lambda x: x-cvals, name="ContentError")(tfeats[0])
    serr = Lambda(lambda x: x-svals, name="StyleError")(tfeats[1])

    model = Model(inputs=[dummy],outputs=[timg,cerr,serr])
    return model


# Begin actual functionality

if len(sys.argv) < 3:
    print("Usage: single <content_img> <style_img>")

simage = load_img(sys.argv[2])
simage = img_to_array(simage)
#simage.show()

cimage = load_img(sys.argv[1])
cimage = img_to_array(cimage)
cdim = cimage.shape
#cimage.show()

print("Building model")
model = single_image_transform_model(cimage,simage)

model.compile(optimizer=optimizers.adam(learning_rate=5.0), 
              loss={ 'ContentError': mean_squared_value, 'StyleError': mean_squared_value, 'result':total_variation_loss },
              loss_weights={'ContentError': 3.0, 'StyleError': 30.0, 'result':1.0})

print("Model built")
#model.summary()

# Initialize result with content image instead of random noise
#model.get_layer('result').set_weights([cimage])

n_iterations = 100
done_epochs = 0
for i in range(n_iterations):
	N = 1000
	model.fit([np.ones((N,)+cdim)],[np.zeros(N)]*3,batch_size=1,
		epochs=done_epochs+1, initial_epoch=done_epochs)

	# Display result
	timg = model.get_layer('result').get_weights()[0]
	timg = timg.reshape(cdim)
	timg = np.minimum(255,np.maximum(0,np.round(timg)))
	timg = array_to_img(timg)
	timg.show()
	done_epochs += 1
