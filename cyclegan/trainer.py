from .utils import *
from .model import *
import tensorflow as tf
import time

import shutil
import os


class trainer(object):
    def __init__(self, args):
        self.output_dir = args['output_dir']
        self.run_name = args['run_name']
        self.sample_dir = os.path.join(self.output_dir, self.run_name, 'samples')
        self.data_file_path = args['data_file_path']
        self.param_file_path = args['param_file_path']
        self.model_param, self.sampler_param, self.training_param = get_config_from_yaml(self.param_file_path)
        self.albumentations_path = args['albumentations_path']



    def train(self):
        """Train cyclegan"""
        self._create_save_dirs()
        generator = get_generator_from_config(self.sampler_param,
                                              data_config_path=self.data_file_path,
                                              albumentations_path=self.albumentations_path,
                                              batch_size=self.training_param['batch_size'])

        start_time = time.time()

        adv_loss_fn = tf.keras.losses.MeanSquaredError()

        # Define the loss function for the generators
        def generator_loss_fn(fake):
            fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
            return fake_loss

        # Define the loss function for the discriminators
        def discriminator_loss_fn(real, fake):
            real_loss = adv_loss_fn(tf.ones_like(real), real)
            fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
            return (real_loss + fake_loss) * 0.5

        # Get the generators
        gen_G = get_unet_generator(input_shape=(None, None, 3),
                                   residual=self.model_param['residual'],
                                   filter_growth=self.model_param['filter_growth'],
                                   depth=self.model_param['depth'],
                                   name = "generator_G")
        gen_F = get_unet_generator(input_shape=(None, None, 3),
                                   residual=self.model_param['residual'],
                                   filter_growth=self.model_param['filter_growth'],
                                   depth=self.model_param['depth'],
                                   name="generator_F")

        # Get the discriminators
        disc_X = get_discriminator(input_shape=(None, None, 3),
                                   filters=self.model_param['disc_filters'],
                                   name="discriminator_X")
        disc_Y = get_discriminator(input_shape=(None, None, 3),
                                   filters=self.model_param['disc_filters'],
                                   name="discriminator_Y")

        cycle_gan_model = CycleGan(
            generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y,
            l_cycle=self.training_param['cycle_lambda'],
            l_ssim=self.training_param['ssim_lambda'],
            l_id=self.training_param['id_lambda']
        )

        cycle_gan_model.compile(
            gen_G_optimizer=tf.keras.optimizers.Adam(learning_rate=self.training_param['learning_rate'], beta_1=0.5),
            gen_F_optimizer=tf.keras.optimizers.Adam(learning_rate=self.training_param['learning_rate'], beta_1=0.5),
            disc_X_optimizer=tf.keras.optimizers.Adam(learning_rate=self.training_param['learning_rate'], beta_1=0.5),
            disc_Y_optimizer=tf.keras.optimizers.Adam(learning_rate=self.training_param['learning_rate'], beta_1=0.5),
            gen_loss_fn=generator_loss_fn,
            disc_loss_fn=discriminator_loss_fn
        )
        callbacks = []
        scheduler = self._get_scheduler(self.training_param['learning_rate'], self.training_param['decay_epoch'])
        callbacks.append(GANLRScheduler(scheduler, verbose=1))
        callbacks.append(GANMonitor(*generator[0], output_path=os.path.join(self.output_dir, self.run_name, 'samples')))
        callbacks.append(GANSaveModels(os.path.join(self.output_dir, self.run_name, 'checkpoint')))

        cycle_gan_model.fit(generator,
                            epochs=self.training_param['epochs'],
                            callbacks=callbacks,
                            workers=4,
                            use_multiprocessing=False,
                            verbose=1)

        print(f"training finished in: {time.time() - start_time}")

    def _create_save_dirs(self):
        config_dir = os.path.join(self.output_dir, self.run_name, 'config')
        checkpoint_dir = os.path.join(self.output_dir, self.run_name, 'checkpoint')
        sample_dir = os.path.join(self.output_dir, self.run_name, 'samples')
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        shutil.copyfile(self.param_file_path, os.path.join(config_dir, os.path.basename(self.param_file_path)))
        shutil.copyfile(self.data_file_path, os.path.join(config_dir, os.path.basename(self.data_file_path)))

    def _get_scheduler(self, learning_rate, decay_epoch):
        def scheduler(epoch, lr):
            new_lr = learning_rate * 0.5 ** (epoch // decay_epoch)
            return new_lr
        return scheduler

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K

class GANLRScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, schedule, verbose=0):
        super(GANLRScheduler, self).__init__(schedule, verbose=verbose)

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(K.get_value(self.model.optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (ops.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        K.set_value(self.model.gen_G_optimizer.lr, K.get_value(lr))
        K.set_value(self.model.gen_F_optimizer.lr, K.get_value(lr))
        K.set_value(self.model.disc_X_optimizer.lr, K.get_value(lr))
        K.set_value(self.model.disc_Y_optimizer.lr, K.get_value(lr))
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))

class GANSaveModels(tf.keras.callbacks.Callback):
    def __init__(self, output_path=None):
        super().__init__()
        self._output_path = output_path

    def on_epoch_begin(self, epoch, logs=None):
        gen_G_path = os.path.join(self._output_path, "source_to_target.h5")
        gen_F_path = os.path.join(self._output_path, "target_to_source.h5")
        self.model.gen_G.save(gen_G_path, overwrite=True)
        self.model.gen_F.save(gen_F_path, overwrite=True)

class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, source_batch, target_batch, output_path):
        super().__init__()
        self.source_batch = source_batch
        self.target_batch = target_batch
        self.batch_size = source_batch.shape[0]
        self.output_path = output_path

    def on_epoch_begin(self, epoch, logs=None):

        img_h = self.source_batch.shape[1]
        img_w = self.source_batch.shape[2]
        output = np.empty((self.batch_size*2*img_h, 3*img_w, 3))
        for i, source in enumerate(self.source_batch):
            source_fake = self._predict_img(source, self.model.gen_G)
            source_cycle = self._predict_img(source_fake / 127.5 - 1, self.model.gen_F)

            orig = np.asarray(source*127.5+127.5, dtype=np.uint8)
            output[i * img_h:i * img_h + img_h, 0:img_w, :] = orig
            output[i * img_h:i * img_h + img_h, img_w:img_w*2, :] = source_fake
            output[i * img_h:i * img_h + img_h, img_w*2:img_w*3, :] = source_cycle

        for i, target in enumerate(self.target_batch):
            target_fake = self._predict_img(target, self.model.gen_F)
            target_cycle = self._predict_img(target_fake / 127.5 - 1, self.model.gen_G)
            i += self.batch_size

            orig = np.asarray(target * 127.5 + 127.5, dtype=np.uint8)
            output[i * img_h:i * img_h + img_h, 0:img_w, :] = orig
            output[i * img_h:i * img_h + img_h, img_w:img_w*2, :] = target_fake
            output[i * img_h:i * img_h + img_h, img_w*2:img_w*3, :] = target_cycle

        output = tf.keras.preprocessing.image.array_to_img(output)
        output.save(os.path.join(self.output_path, "generated_img_{epoch}.png".format(epoch=epoch + 1)))

    def _predict_img(self, img, model):
        prediction = model(img[None])[0].numpy()
        prediction = prediction * 127.5 + 127.5
        prediction = np.clip(prediction, 0.0, 255.0)
        return np.asarray(prediction, dtype=np.uint8)