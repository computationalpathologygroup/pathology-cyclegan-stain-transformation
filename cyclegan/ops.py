import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from .utils import *

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-6, scale=True, scope=name)

def instance_norm(input, name="instance_norm", epsilon=1e-5):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.truncated_normal_initializer(1.0, 0.01, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        normalized = (input-mean) / ((variance + epsilon) ** 0.5)
        return scale*normalized + offset

def spectral_norm(w, iteration=1):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(0.05), trainable=False)

   u_hat = u
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = tf.nn.l2_normalize(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = tf.nn.l2_normalize(u_)

   u_hat = tf.stop_gradient(u_hat)
   v_hat = tf.stop_gradient(v_hat)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = w / sigma
       w_norm = tf.reshape(w_norm, w_shape)


   return w_norm

def conv2d(input_, output_dim, ks=4, s=2, spec_norm=False, stddev=0.01, padding='SAME', name="conv2d", reg=0.01):
    if padding == "REFLECT":
        x = tf.pad(input_, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        p = "VALID"
    else:
        x = input_
        p = padding
    with tf.variable_scope(name):
        w = tf.get_variable("kernel", shape=[ks, ks, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if spec_norm:
            y = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, s, s, 1], padding=p, name=name)
        else:
            y = tf.nn.conv2d(input=x, filter=w, strides=[1, s, s, 1], padding=p, name=name)

        return y

# def conv2d(input_, output_dim, ks=4, s=2, stddev=0.001, padding='SAME', name="conv2d", reg=0.01):
#     if padding == "REFLECT":
#         x = tf.pad(input_, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
#         p = "VALID"
#     else:
#         x = input_
#         p = padding
#     with tf.variable_scope(name):
#         y = tf.layers.conv2d(inputs=x,
#                              filters=output_dim,
#                              kernel_size=ks,
#                              strides=s,
#                              padding=p,
#                              use_bias=False,
#                              kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
#         return y

def maxpool(input, kernel_size=[2, 2], name="maxpool"):
    with tf.variable_scope(name):
        return tf.layers.max_pooling2d(input, pool_size=kernel_size, strides=2)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.01, name="deconv2d"):
    with tf.variable_scope(name):
        return tf.layers.conv2d_transpose(input_, output_dim, ks, s, padding='SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                          bias_initializer=None)

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

def upsample2d(_input, ratio=2, method=ResizeMethod.BILINEAR, align_corners=True, name="upsampling2d"):
    shape = tf.shape(_input)
    new_shape = [shape[1] * ratio, shape[2] * ratio]
    with tf.variable_scope(name):
        return tf.image.resize_images(_input, new_shape, method=method, align_corners=align_corners)


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def crop_layer(layer_to_crop, target_layer):
    src_shape = tf.shape(layer_to_crop)
    target_shape = tf.shape(target_layer)
    x, y = [(src_shape[1] - target_shape[1]) // 2, (src_shape[2] - target_shape[2]) // 2]
    return layer_to_crop[:,x:-x,y:-y,:]