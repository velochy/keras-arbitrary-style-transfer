from keras.models import Model,Sequential,load_model,model_from_json
from keras.layers import *
import numpy as np
import keras.backend as K
from keras import regularizers, activations
from custom_layers import ImgPreprocess, InstanceNormWithScalingInputs, InstanceNorm, CovarianceMatrix

def NormD(act,nw,axis,**kwargs):
    dsize = act._keras_shape[axis]
    return InstanceNormWithScalingInputs(axis=axis,**kwargs)([act,
                Dense(dsize, activation='linear')(nw),
                Dense(dsize, activation='linear')(nw)])


def transfer_model(idim, style_n):

    img_inp = Input(shape=idim,dtype="float32",name="img")
    style_inp = Input(shape=(style_n,),dtype="float32",name="style")

    conv_p = { 'padding': 'same', 'activation': 'relu'}
    conv_l = { 'padding': 'same', 'activation': 'linear'}

    x = NormD(Conv2D(32,(9,9),**conv_p)(img_inp), style_inp, 3)
    x = NormD(Conv2D(64,(3,3),strides=(2,2),**conv_p)(x), style_inp, 3)
    x = NormD(Conv2D(128,(3,3),strides=(2,2),**conv_p)(x), style_inp, 3)

    x2 = NormD(Conv2D(128,(3,3),**conv_p)(x), style_inp, 3)
    x3 = NormD(Conv2D(128,(3,3),**conv_l)(x2), style_inp, 3)
    x = Activation('relu')(Add()([x,x3]))

    x2 = NormD(Conv2D(128,(3,3),**conv_p)(x), style_inp, 3)
    x3 = NormD(Conv2D(128,(3,3),**conv_l)(x2), style_inp, 3)
    x = Activation('relu')(Add()([x,x3]))

    x2 = NormD(Conv2D(128,(3,3),**conv_p)(x), style_inp, 3)
    x3 = NormD(Conv2D(128,(3,3),**conv_l)(x2), style_inp, 3)
    x = Activation('relu')(Add()([x,x3]))

    x2 = NormD(Conv2D(128,(3,3),**conv_p)(x), style_inp, 3)
    x3 = NormD(Conv2D(128,(3,3),**conv_l)(x2), style_inp, 3)
    x = Activation('relu')(Add()([x,x3]))

    x2 = NormD(Conv2D(128,(3,3),**conv_p)(x), style_inp, 3)
    x3 = NormD(Conv2D(128,(3,3),**conv_l)(x2), style_inp, 3)
    x = Activation('relu')(Add()([x,x3]))

    x2 = UpSampling2D((2,2))(x)
    x = NormD(Conv2D(64,(3,3),**conv_p)(x2), style_inp, 3)

    x2 = UpSampling2D((2,2))(x)
    x = NormD(Conv2D(32,(3,3),**conv_p)(x2), style_inp, 3)

    img_outp = Conv2D(3,(9,9),padding='same',activation='sigmoid', name="ImgO")(x)

    model = Model(inputs=[img_inp, style_inp],outputs=[img_outp])

    return model

def style_model(idim, style_n):

    img_inp = Input(shape=idim,dtype="float32",name="img")

    pp_img_inp = ImgPreprocess()(img_inp)

    from keras.applications.inception_v3 import InceptionV3
    imodel = InceptionV3(input_tensor=pp_img_inp,include_top=False)

    #from keras.applications.mobilenet_v2 import MobileNetV2
    #imodel = MobileNetV2(input_tensor=pp_img_inp,include_top=False)

    #imodel.trainable = False
    
    x = GlobalAveragePooling2D()(imodel.outputs[0])
    style = Dense(style_n, activation="sigmoid", name='Style')(x)

    model = Model(inputs=[img_inp],outputs=[style])
    return model

def StyleMetric(act):
    x = CovarianceMatrix(axis=3)(act)
    x = InstanceNorm()(Flatten()(x))
    return x


def extract_features(idim):
    img_inp = Input(shape=idim,dtype="float32",name="img")

    pp_img_inp = ImgPreprocess()(img_inp)

    from keras.applications.vgg16 import VGG16
    imodel = VGG16(input_tensor=pp_img_inp,include_top=False)

    content_layers = ['block4_conv1']
    if len(content_layers)>1:
        contents = list(map(lambda n: Flatten()(InstanceNorm()(imodel.get_layer(n).output)), content_layers))
        content = Concatenate(name="Content")(contents) 
    else:
        content = Flatten(name="Content")(InstanceNorm()(imodel.get_layer(content_layers[0]).output))

    #style_layers = ['block1_conv1','block1_conv2','block2_conv1','block2_conv2','block3_conv1','block3_conv2','block3_conv3','block4_conv1','block4_conv2','block4_conv3','block5_conv1','block5_conv2','block5_conv3']
    style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1']
    styles = map(lambda n: StyleMetric(imodel.get_layer(n).output), style_layers)
    style = Concatenate(name="Style")(list(styles))

    model = Model(inputs=[img_inp],outputs=[content,style])
    return model

def full_model(cdim, sdim, n_style):
    smodel = style_model(sdim,n_style)
    #smodel.summary()
    tmodel = transfer_model(cdim,n_style)
    #tmodel.summary()
    
    fcmodel = extract_features(cdim)
    fcmodel.trainable = False
    #fsmodel.summary()

    if cdim==sdim: fsmodel = fcmodel
    else:
        fsmodel = extract_features(sdim)
        fsmodel.trainable = False

    cimg_inp = Input(shape=cdim,dtype="float32",name="cimg")
    simg_inp = Input(shape=sdim,dtype="float32",name="simg")

    style = smodel([simg_inp])
    timg = tmodel([cimg_inp,style])
    tfeats = fcmodel(timg)

    cerr = Subtract(name="ContentError")([tfeats[0],fcmodel(cimg_inp)[0]])
    serr = Subtract(name="StyleError")([tfeats[1],fsmodel(simg_inp)[1]])

    model = Model(inputs=[cimg_inp,simg_inp],outputs=[cerr,serr])
    return model
