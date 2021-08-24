from __future__ import division
from .utils import *
import tensorflow as tf
from .ops import *
import tensorflow_probability as tfp



# noinspection PyMethodOverriding
class CycleGan(tf.keras.Model):
    def __init__(self, generator_G, generator_F, discriminator_X, discriminator_Y, l_cycle=10.0, l_ssim=0.3, l_id=1.0):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = l_cycle
        self.lambda_id = l_id
        self.lambda_ssim = l_ssim

    def compile(self, gen_G_optimizer, gen_F_optimizer, disc_X_optimizer, disc_Y_optimizer, gen_loss_fn, disc_loss_fn,
                **kwargs):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()

    def call(self, inputs):
        print("The call function is not implemented for cyclegan and should thus not be used")
        return inputs

    # def cycle_loss_fn(self, real, cycle):
    #     abs_ = tf.math.abs(real - cycle)
    #     percentile_ = tfp.stats.percentile(abs_, q=90.)
    #     return tf.math.reduce_mean(tf.boolean_mask(abs_, tf.math.greater(abs_, percentile_)))

    def ssim_loss(self, real, fake):
        return tf.math.reduce_mean(tf.image.ssim(real, fake, 2.0))

    def train_step(self, batch_data):
        real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:

            fake_y = self.gen_G(real_x, training=True)
            fake_x = self.gen_F(real_y, training=True)

            cycled_x = self.gen_F(fake_y, training=True)
            cycled_y = self.gen_G(fake_x, training=True)

            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            id_loss_G = self.identity_loss_fn(real_y, same_y) * self.lambda_id
            id_loss_F = self.identity_loss_fn(real_x, same_x) * self.lambda_id

            ssim_loss_G = self.ssim_loss(real_x, fake_y) * self.lambda_ssim
            ssim_loss_F = self.ssim_loss(real_y, fake_x) * self.lambda_ssim

            total_loss_G = gen_G_loss + cycle_loss_G + ssim_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + ssim_loss_F + id_loss_F

            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators

        self.gen_G_optimizer.apply_gradients(zip(grads_G, self.gen_G.trainable_variables))
        self.disc_X_optimizer.apply_gradients(zip(disc_X_grads, self.disc_X.trainable_variables))

        self.gen_F_optimizer.apply_gradients(zip(grads_F, self.gen_F.trainable_variables))
        self.disc_Y_optimizer.apply_gradients(zip(disc_Y_grads, self.disc_Y.trainable_variables))

        return {
            "disc_G_loss": gen_G_loss,
            "disc_F_loss": gen_F_loss,
            "cyc_G_loss": cycle_loss_G,
            "cyc_F_loss": cycle_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }

def get_unet_generator(input_shape, residual, filter_growth, depth, name=None):
    """
    Recursively generate a U-net.
    """
    input = tf.keras.layers.Input(shape=input_shape, name=name + "_img_input")
    x = unet_block(input, 0, filter_growth, depth)
    x = tf.keras.layers.Conv2D(3, (1,1), kernel_initializer=kernel_init, use_bias=False)(x)
    if residual:
        output =  2 * tf.keras.layers.Activation("tanh")(x) + input
    else:
        output = tf.keras.layers.Activation("tanh")(x)
    model = tf.keras.models.Model(input, output, name=name)
    return model

def get_discriminator(input_shape, filters, name=None):

    img_input = tf.keras.layers.Input(shape=input_shape, name=name + "_img_input")
    kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)

    x = tf.keras.layers.Conv2D(filters, (4, 4), strides=(2, 2), padding="same",
                               kernel_initializer=kernel_initializer)(img_input)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(filters*2, (4, 4), strides=(2, 2), padding="same",
                               kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init, epsilon=1e-5)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(filters*4, (4, 4), strides=(2, 2), padding="same",
                               kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init, epsilon=1e-5)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(filters*4, (4, 4), strides=(2, 2), padding="same",
                               kernel_initializer=kernel_initializer)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init, epsilon=1e-5)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same",
                               kernel_initializer=kernel_initializer)(x)
    model = tf.keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model
