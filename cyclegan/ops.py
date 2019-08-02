import math
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from cyclegan.utils import *

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.01, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        normalized = (input-mean) / (variance + epsilon) ** 0.5
        return scale*normalized + offset

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.01, padding='SAME', name="conv2d", reg=0.0):
    if padding == "REFLECT":
        x = tf.pad(input_, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        p = "VALID"
    else:
        x = input_
        p = padding
    with tf.variable_scope(name):
        y = tf.layers.conv2d(inputs=x,
                             filters=output_dim,
                             kernel_size=ks,
                             strides=s,
                             padding=p,
                             use_bias=False,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                             kernel_regularizer=tf.keras.regularizers.l2(reg))
        return y

def maxpool(input, kernel_size=[2, 2], name="maxpool"):
    with tf.variable_scope(name):
        return tf.layers.max_pooling2d(input, pool_size=kernel_size, strides=2)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return tf.layers.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)

def downsample_bilinear(_input, ratio=2, align_corners=True, name="downsampling2d_bilinear"):
    shape = tf.shape(_input)
    new_shape = [shape[1] // ratio, shape[2] // ratio]
    with tf.variable_scope(name):
        return tf.image.resize_images(_input, new_shape, align_corners=align_corners)

# def downsample_bilinear(_input, ratio=2, align_corners=True, name="downsampling2d_bilinear"):
#     shape = tf.cast(tf.shape(_input),dtype=tf.float32)
#     new_shape = [shape[1] * ratio, shape[2] * ratio]
#     with tf.variable_scope(name):
#         return tf.image.resize_bilinear(_input, tf.cast(tf.round(new_shape), dtype=tf.int32), align_corners=align_corners)

def upsample2d(_input, ratio=2, method=ResizeMethod.NEAREST_NEIGHBOR, align_corners=True, name="upsampling2d"):
    shape = tf.shape(_input)
    new_shape = [shape[1] * ratio, shape[2] * ratio]
    with tf.variable_scope(name):
        return tf.image.resize_images(_input, new_shape, method=method, align_corners=align_corners)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def crop_layer(layer_to_crop, target_layer):
    down_shape = tf.shape(layer_to_crop)
    up_shape = tf.shape(target_layer)
    x, y = [(down_shape[1] - up_shape[1]) // 2, (down_shape[2] - up_shape[2]) // 2]
    return layer_to_crop[:,x:-x,y:-y,:]