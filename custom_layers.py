from keras.layers import Layer, InputSpec
from keras import backend as K

class InstanceNorm(Layer):
    """Instance normalization"""
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self.supports_masking = False
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        ndim = len(input_shape)
        self.input_spec = InputSpec(ndim=ndim)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        reduction_axes = list(range(0, len(input_shape)))
        if self.axis is not None:
            del reduction_axes[self.axis]
        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev
    
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
        }
        base_config = super(InstanceNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class InstanceNormWithScalingInputs(Layer):
    """Instance normalization with beta and gamma weights given as inputs
    # Arguments
        axis: The axis that should be normalized
            (typically the features axis).
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
    # Input shape - three inputs: one is the inputs to normalize, the second is gamma and third is beta
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 **kwargs):
        super(InstanceNormWithScalingInputs, self).__init__(**kwargs)
        self.supports_masking = False
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        ndim = len(input_shape[0])
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        if self.axis is None:
            shape = (None,1,)
        else:
            shape = (None,input_shape[0][self.axis],)            

        self.input_spec = [InputSpec(ndim=ndim),InputSpec(shape=shape),InputSpec(shape=shape)]

        self.built = True

    def call(self, inputs_list, training=None):
        [inputs, gamma, beta] = inputs_list
        input_shape = K.int_shape(inputs)

        reduction_axes = list(range(0, len(input_shape)))
        if self.axis is not None:
            del reduction_axes[self.axis]
        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[0] = -1 # Batch dimension
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        broadcast_gamma = K.reshape(gamma, broadcast_shape)
        normed = normed * broadcast_gamma
    
        broadcast_beta = K.reshape(beta, broadcast_shape)
        normed = normed + broadcast_beta
    
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
        }
        base_config = super(InstanceNormWithScalingInputs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

""" Gram matrix alone can be calculated with existing layers using:
    lastdim = act._keras_shape[-1]
    x = Reshape((-1,lastdim))(act)
    dot_dim = x._keras_shape[1]
    px = Dot(axes=1)([x,x])
    x = Lambda(lambda z: z/dot_dim)(x)
    But it is hacky so a proper layer is probably better
"""
class CovarianceMatrix(Layer):
    """ Compute the Covariance matrix (which is slightly more advanced from simple Gram matrix)
    """
    def __init__(self, axis=3,
                 **kwargs):
        super(CovarianceMatrix, self).__init__(**kwargs)
        self.supports_masking = False
        self.axis = axis

    def build(self, input_shape):
        ndim = len(input_shape)
        self.input_spec = InputSpec(ndim=ndim)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        reduction_axes = list(range(0, len(input_shape)))
        if self.axis is not None:
            del reduction_axes[self.axis]
        del reduction_axes[0]

        # Put axis last
        inputs = K.permute_dimensions(inputs, tuple( [0] + reduction_axes + [self.axis]))

        # Collapse all other dims into dim 1
        cinp = K.reshape(inputs, (K.shape(inputs)[0],-1,input_shape[self.axis]))
        n_reduced = K.shape(cinp)[1]

        # Calculate dot product
        pure_gram = K.batch_dot(cinp,cinp,1)
        scaled_gram = pure_gram/K.cast(n_reduced,'float32')
        
        # Calculate covariance
        means = K.mean(cinp, [1], keepdims=True)
        mean_mat = K.batch_dot(means,means,1)
        cov = scaled_gram - mean_mat

        return cov

    # As the automatic inference does not work here, unfortunately
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[self.axis],input_shape[self.axis])

    def get_config(self):
        config = { 'axis': self.axis }
        base_config = super(CovarianceMatrix, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ImgPreprocess(Layer):
    """Do the preprocessing used for VGG and MobileNetV2 pretrained weights
        Or, at least, close enough
    """
    def __init__(self, axis=3,
                 **kwargs):
        super(ImgPreprocess, self).__init__(**kwargs)
        self.supports_masking = False
        self.axis = axis

    def build(self, input_shape):
        ndim = len(input_shape)
        self.input_spec = InputSpec(ndim=ndim)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        reduction_axes = list(range(0, len(input_shape)))
        if self.axis is not None:
            del reduction_axes[self.axis]
        del reduction_axes[0]

        inputs = K.reverse(inputs,axes=self.axis) # RGB to BGR

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        normed = 255.0*(inputs - mean)

        return normed

    def get_config(self):
        config = { 'axis': self.axis }
        base_config = super(ImgPreprocess, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Metric that tries to minimize the length of y_pred
def mean_squared_value(y_true,y_pred):
    return K.mean(K.square(y_pred))

CUSTOM_OBJECTS = { 'InstanceNorm': InstanceNorm, 'InstanceNormWithScalingInputs': InstanceNormWithScalingInputs,
                    'ImgPreprocess': ImgPreprocess,'mean_squared_value': mean_squared_value, 'CovarianceMatrix':CovarianceMatrix }