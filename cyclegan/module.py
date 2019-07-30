from __future__ import division
import tensorflow as tf
from cyclegan.ops import *
from cyclegan.utils import *

# def discriminator(image, options, reuse=False, name="discriminator"):
#     with tf.variable_scope(name):
#         # image is 256 x 256 x input_c_dim
#         if reuse:
#             tf.get_variable_scope().reuse_variables()
#         else:
#             assert tf.get_variable_scope().reuse is False
#
#         h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
#         # h0 is (128 x 128 x self.df_dim)
#         h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 2, name='d_h1_conv'), 'd_bn1'))
#         # h1 is (64 x 64 x self.df_dim*2)
#         h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 4, name='d_h2_conv'), 'd_bn2'))
#         # h2 is (32x 32 x self.df_dim*4)
#         h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 8, s=1, name='d_h3_conv'), 'd_bn3'))
#         # h3 is (32 x 32 x self.df_dim*8)
#         h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
#         # h4 is (32 x 32 x 1)
#         return h4

def discriminator(image, options, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 4, name='d_h3_conv'), 'd_bn3'))
        # h2 is (16x 16 x self.df_dim*4)
        h4 = lrelu(instance_norm(conv2d(h3, options.df_dim * 4, name='d_h4_conv'), 'd_bn4'))
        h4_1 = lrelu(instance_norm(conv2d(h3, options.df_dim * 8, s=1, name='d_h41_conv'), 'd_bn41'))
        h4_output = conv2d(h4_1, 1, s=1, name='d_h4_pred')
        # h2 is (8x 8 x self.df_dim*4)
        h5 = lrelu(instance_norm(conv2d(h4, options.df_dim * 4, name='d_h5_conv'), 'd_bn5'))
        # h2 is (4x 4 x self.df_dim*4)
        h6 = lrelu(instance_norm(conv2d(h5, options.df_dim * 8, s=1, name='d_h6_conv'), 'd_bn6'))
        # h3 is (4 x 4 x self.df_dim*8)
        h7_output = conv2d(h6, 1, s=1, name='d_h7_pred')
        # h4 is (4 x 4 x 1)
        return h4_output, h7_output

def conv_block(_input, level, filters, padding, residual=False, regularization=0.0):
    name1 = 'g_d{}_conv1'.format(level)
    name2 = 'g_d{}_conv2'.format(level)
    net = lrelu(instance_norm(conv2d(_input, output_dim=filters, ks=3, s=1, padding=padding, name=name1,
                                     reg=regularization),
                        name='g_d{}_in1'.format(level)))
    tf.summary.histogram("conv1_{}".format(level), net)
    if residual and level > 0:
        net = lrelu(instance_norm(conv2d(net,
                                         output_dim=_input.shape[3].value,
                                         ks=3,
                                         s=1,
                                         padding=padding,
                                         name='g_d{}_res'.format(level),
                                         reg=regularization),
                                  name='g_d{}_in2'.format(level)))
        net = _input + net
    else:
        net = lrelu(instance_norm(conv2d(net, output_dim=filters, ks=3, s=1, padding=padding, name=name2, reg=regularization),
                                  name='g_d{}_in2'.format(level)))
        tf.summary.histogram("conv2_{}".format(level), net)
    return net


def unet_block(_input, level, options):
    filters = options.gf_dim * 2 ** level
    if options.unet_depth - level > 0:
        with tf.variable_scope("down"):
            down_block = conv_block(_input, level, filters=filters, padding=options.padding,
                                    residual=options.unet_residual, regularization=options.regularization)
        if options.use_maxpool:
            net = maxpool(down_block, name='g_d{}_maxpool'.format(level))
        else:
            net = downsample_bilinear(down_block)
        net = unet_block(net, level + 1, options)
        net = upsample2d(net, name='g_d{}_upsample2d'.format(level))
        net = tf.concat([down_block, net], 3)
        # net = tf.concat([down_block, instance_norm(net, name='g_d{}_deconv_bn'.format(level))], 3)
        with tf.variable_scope("up"):
            net = conv_block(net, level, filters=filters, padding=options.padding, residual=options.unet_residual,
                             regularization=options.regularization)
    else:
        net = conv_block(_input, level, filters=filters, padding=options.padding, residual=options.unet_residual,
                         regularization=options.regularization)
    return net


def generator_unet(image, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        net = unet_block(image, 0, options)
        output = conv2d(net, options.output_c_dim, ks=1, s=1, padding='VALID', name='g_conv_final')
        return tf.nn.tanh(output) + image


# def generator_resnet(image, options, reuse=False, name="generator"):
#     with tf.variable_scope(name):
#         # image is 256 x 256 x input_c_dim
#         if reuse:
#             tf.get_variable_scope().reuse_variables()
#         else:
#             assert tf.get_variable_scope().reuse is False
#
#         def residual_block(x, dim, ks=3, s=1, _name='res'):
#             p = int((ks - 1) / 2)
#             y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
#             y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=_name + '_c1'), _name + '_bn1')
#             y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
#             y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=_name + '_c2'), _name + '_bn2')
#             return y + x
#
#         # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
#         # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
#         # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
#         c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
#         c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
#         c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
#         c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
#         # define G network with 9 resnet blocks
#         r1 = residual_block(c3, options.gf_dim * 4, _name='g_r1')
#         r2 = residual_block(r1, options.gf_dim * 4, _name='g_r2')
#         r3 = residual_block(r2, options.gf_dim * 4, _name='g_r3')
#         r4 = residual_block(r3, options.gf_dim * 4, _name='g_r4')
#         r5 = residual_block(r4, options.gf_dim * 4, _name='g_r5')
#         r6 = residual_block(r5, options.gf_dim * 4, _name='g_r6')
#         r7 = residual_block(r6, options.gf_dim * 4, _name='g_r7')
#         r8 = residual_block(r7, options.gf_dim * 4, _name='g_r8')
#         r9 = residual_block(r8, options.gf_dim * 4, _name='g_r9')
#
#         d1 = deconv2d(r9, options.gf_dim * 2, 3, 2, name='g_d1_dc')
#         d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
#         d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
#         d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
#         d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
#         pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))
#
#         return pred


def focal_abs_criterion(in_,target, percentile=90.0):
    abs_ = tf.abs(in_ - target)
    percentile_ = tf.contrib.distributions.percentile(abs_, q=90.)
    return tf.reduce_mean(tf.boolean_mask(abs_, tf.greater(abs_, percentile_)))


def abs_criterion(in_, target, padding="SAME"):
    # if padding == "VALID":
    #     in_ = crop_layer(in_, target)
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    if target:
        return tf.reduce_mean((in_[0] - tf.ones_like(in_[0])) ** 2) + tf.reduce_mean((in_[1] - tf.ones_like(in_[1])) ** 2)
    else:
        return tf.reduce_mean((in_[0] - tf.zeros_like(in_[0])) ** 2) + tf.reduce_mean((in_[1] - tf.zeros_like(in_[1])) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
