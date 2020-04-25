from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import *
import numpy as np
import tensorflow.keras.backend as K
from keras import regularizers, activations
from custom_layers import ImgPreprocess, InstanceNormWithScalingInputs, InstanceNorm, GramMatrix, RawWeights, total_variation_loss

def NormD(act,nw,axis=3,**kwargs):
    dsize = act._keras_shape[axis]
    
    return  InstanceNormWithScalingInputs(axis=axis,**kwargs)([act,
                Dense(dsize, activation=None)(nw),
                Dense(dsize, activation=None)(nw)])

def NormDA(act,nw,axis=3):
    # NB! Activation *after* normalization
    return Activation('relu')(NormD(act,nw,axis))


def transfer_model(idim, style_n):

    img_inp = Input(shape=idim,dtype="float32",name="img")
    style_inp = Input(shape=(style_n,),dtype="float32",name="style")

    # Activation done inside NormD after normalization,
    # and as normalization subtracts mean, bias becomes useless
    conv_p = { 'padding': 'same', 'activation': None, 'use_bias': False}

    x = NormDA(SeparableConv2D(32,(9,9),**conv_p)(img_inp), style_inp)
    x = NormDA(SeparableConv2D(64,(3,3),strides=(2,2),**conv_p)(x), style_inp)
    x = NormDA(SeparableConv2D(128,(3,3),strides=(2,2),**conv_p)(x), style_inp)

    # Residual block
    x2 = NormDA(SeparableConv2D(128,(3,3),**conv_p)(x), style_inp)
    x2 = NormD(SeparableConv2D(128,(3,3),**conv_p)(x2), style_inp)
    x = Activation('relu')(Add()([x,x2]))

    # Residual block
    x2 = NormDA(SeparableConv2D(128,(3,3),**conv_p)(x), style_inp)
    x2 = NormD(SeparableConv2D(128,(3,3),**conv_p)(x2), style_inp)
    x = Activation('relu')(Add()([x,x2]))

    # Residual block
    x2 = NormDA(SeparableConv2D(128,(3,3),**conv_p)(x), style_inp)
    x2 = NormD(SeparableConv2D(128,(3,3),**conv_p)(x2), style_inp)
    x = Activation('relu')(Add()([x,x2]))

    # Residual block
    x2 = NormDA(SeparableConv2D(128,(3,3),**conv_p)(x), style_inp)
    x2 = NormD(SeparableConv2D(128,(3,3),**conv_p)(x2), style_inp)
    x = Activation('relu')(Add()([x,x2]))

    # Residual block
    x2 = NormDA(SeparableConv2D(128,(3,3),**conv_p)(x), style_inp)
    x2 = NormD(SeparableConv2D(128,(3,3),**conv_p)(x2), style_inp)
    x = Activation('relu')(Add()([x,x2]))

    x2 = UpSampling2D((2,2),interpolation='bilinear')(x)
    x = NormDA(SeparableConv2D(64,(3,3),**conv_p)(x2), style_inp)

    x2 = UpSampling2D((2,2),interpolation='bilinear')(x)
    x = NormDA(SeparableConv2D(32,(3,3),**conv_p)(x2), style_inp)

    img_outp = Conv2D(3,(9,9), padding='same',activation='sigmoid',use_bias='true')(x)

    scaled_img_outp = Lambda(lambda x: 255*x, name="ImgOut")(img_outp)

    model = Model(inputs=[img_inp, style_inp],outputs=[scaled_img_outp])

    return model

def style_model(idim, style_n):

    img_inp = Input(shape=idim,dtype="float32",name="img")

    pp_img_inp = ImgPreprocess()(img_inp)

    #from tensorflow.keras.applications.inception_v3 import InceptionV3
    #imodel = InceptionV3(input_tensor=pp_img_inp,include_top=False,weights='imagenet')
    #out_layer = imodel.get_layer("mixed6")

    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    imodel = MobileNetV2(input_tensor=pp_img_inp,include_top=False,weights='imagenet')
    out_layer = imodel.get_layer('block_14_add')

    imodel.trainable = False
    
    x = GlobalAveragePooling2D()(out_layer.output)
    #x = Dense(style_n, activation="sigmoid")(x)
    style = Dense(style_n, activation='sigmoid', name='Style')(x)

    model = Model(inputs=[img_inp],outputs=[style])
    return model

def StyleMetric(act):
    x = GramMatrix(axis=3)(act)
    x = Flatten()(x)
    x = InstanceNorm()(x)
    return x


def extract_features(idim):
    img_inp = Input(shape=idim,dtype="float32",name="img")

    pp_img_inp = ImgPreprocess()(img_inp)

    from tensorflow.keras.applications.vgg19 import VGG19
    imodel = VGG19(input_tensor=pp_img_inp,pooling='avg',include_top=False,weights='imagenet')

    content_layers = ['block4_conv2']
    if len(content_layers)>1:
        contents = list(map(lambda n: InstanceNorm()(Flatten()(imodel.get_layer(n).output))), content_layers)
        content = Concatenate(name="Content")(contents) 
    else:
        content = InstanceNorm(name="Content")(Flatten()(imodel.get_layer(content_layers[0]).output))

    #style_layers = ['block1_conv1','block1_conv2','block2_conv1','block2_conv2','block3_conv1','block3_conv2','block3_conv3','block4_conv1','block4_conv2','block4_conv3','block5_conv1','block5_conv2','block5_conv3']
    style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1']
    #ScaleL = Lambda(lambda x: x / len(style_layers))
    styles = map(lambda n: StyleMetric(imodel.get_layer(n).output), style_layers)
    style = Concatenate(name="Style")(list(styles))

    model = Model(inputs=[img_inp],outputs=[content,style])
    return model

def full_model(tmodel, smodel):
    cdim, sdim = tmodel.input_shape[0][1:], smodel.input_shape[1:]
    
    tmodel.name, smodel.name = 'model_transfer','model_style'
    
    fcmodel = extract_features(cdim)
    fcmodel.trainable = False

    if cdim==sdim: fsmodel = fcmodel
    else:
        fsmodel = extract_features(sdim)
        fsmodel.trainable = False

    cimg_inp = Input(shape=cdim,dtype="float32",name="cimg")
    simg_inp = Input(shape=sdim,dtype="float32",name="simg")

    style = smodel([simg_inp])
    timg = tmodel([cimg_inp,style])
    res = Lambda(lambda x:x,name="result")(timg)

    tfeats = fcmodel(timg)

    cerr = Subtract(name="ContentError")([tfeats[0],fcmodel(cimg_inp)[0]])
    serr = Subtract(name="StyleError")([tfeats[1],fsmodel(simg_inp)[1]])

    model = Model(inputs=[cimg_inp,simg_inp],outputs=[res,cerr,serr])
    return model


