import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import tensorflow as tf
from .utils import *
import tensorflow_addons as tfa

class ReflectionPadding2D(tf.keras.layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_w, padding_h = self.padding
        padding_tensor = [
            [0, 0],
            [padding_h, padding_h],
            [padding_w, padding_w],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


    def get_config(self):
        config = super().get_config().copy()
        config.update({"padding": self.padding})
        return config


kernel_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

gamma_init = tf.keras.initializers.TruncatedNormal(mean=1.0, stddev=0.01)


def downsample_bilinear(x):
    shape = tf.shape(x)
    new_shape = [shape[1] // 2, shape[2] // 2]
    return tf.image.resize(x, new_shape)

def upsample_bilinear(x):
    shape = tf.shape(x)
    new_shape = [shape[1] * 2, shape[2] * 2]
    return tf.image.resize(x, new_shape)

def unet_block(x, level, filter_growth, depth):
    filters = filter_growth * 2 ** level
    if depth - level > 0:
        down_block = conv_block(x, filters)

        x = downsample_bilinear(down_block)
        x = unet_block(x, level + 1, filter_growth, depth)
        x = upsample_bilinear(x)
        x = tf.keras.layers.concatenate([down_block, x], axis=3)
        x = conv_block(x, filters, up=True)
    else:
        x = conv_block(x, filters)
    return x

def conv_block(x, filters, up=False, kernel_initializer=kernel_init, gamma_initializer=gamma_init):
    x = ReflectionPadding2D()(x)
    x = tf.keras.layers.Conv2D(filters, (3,3), (1,1), kernel_initializer=kernel_initializer, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tfa.layers.GroupNormalization(groups=16, gamma_initializer=gamma_initializer, epsilon=1e-5)(x)

    x = ReflectionPadding2D()(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), (1, 1), kernel_initializer=kernel_initializer, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    if not up:
        x = tfa.layers.GroupNormalization(groups=16, gamma_initializer=gamma_initializer, epsilon=1e-5)(x)
    return x
