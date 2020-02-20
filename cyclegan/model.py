from __future__ import division
import os
import time
from collections import namedtuple
import yaml
import shutil
from .module import *
from .utils import *
import numpy as np

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.output_dir = args['output_dir']
        self.run_name = args['run_name']
        self.sample_dir = os.path.join(self.output_dir, self.run_name, 'samples')
        self.debug_data = args['debug_data']

        self.discriminator = discriminator
        if args['network'] == 'unet':
            self.generator = generator_unet
        elif args['network'] == 'resnet':
            self.generator = generator_resnet
        self.data_file_path = args['data_file_path']
        self.param_file_path = args['param_file_path']
        self._configure_variables()
        if self.use_logloss:
            self.criterionGAN = sce_loss
        else:
            self.criterionGAN = mae_loss
        self._build_model()
        self.saver = tf.train.Saver()

    def _configure_variables(self):
        with open(file=self.param_file_path, mode='r') as param_file:
            parameters = yaml.load(stream=param_file)
        model_params = parameters['model']
        training_params = parameters['training']

        self.spacing = training_params['spacing']
        self.iterations = training_params['iterations']
        self.mini_batch_size = training_params['mini_batch_size']
        self.batch_size = training_params['iterations']['source']['batch size']
        self.L1_lambda = training_params['L1_lambda']
        self.l2_lambda = training_params['l2_lambda']
        self.learning_rate = training_params['learning_rate']
        self.beta1 = training_params['beta1']
        self.lr_decay_epoch = training_params['decay_epoch']
        self.epochs = training_params['epoch count']
        self.id_lambda = training_params['id_lambda']
        self.D_lambda = training_params['D_lambda']
        self.gamma = training_params['gamma']
        self.use_logloss = training_params['use_logloss']
        self.input_c_dim = model_params['input_c_dim']
        self.output_c_dim = model_params['output_c_dim']
        self.normalization = model_params['normalization']
        self._disc_standalone_epochs = training_params['disc_standalone_epochs']
        if model_params['cycle_loss'] == 'focal':
            self.criterion_cycle = focal_abs_loss
        else:
            self.criterion_cycle = abs_loss

        OPTIONS = namedtuple('OPTIONS',
                             'residualgan use_maxpool gf_dim df_dim output_c_dim unet_depth padding unet_residual regularization')
        self.options = OPTIONS._make((model_params['residualgan'], model_params['use_maxpool'], model_params['ngf'], model_params['ndf'],
                                      self.output_c_dim, model_params['unet_depth'], model_params['padding'],
                                      model_params['unet_residual'], model_params['regularization']))



    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, None, None, self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')
        self.identity_lambda = tf.placeholder(tf.float32, None, name="identity_lambda")
        self.D_lambda_var = tf.placeholder(tf.float32, None, name="D_lambda")
        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A = self.generator(self.real_B, self.options, False, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, True, name="generatorB2A")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")

        self.g_loss_a2b = self.D_lambda_var * self.criterionGAN(self.DB_fake, True)
        self.g_loss_b2a = self.D_lambda_var * self.criterionGAN(self.DA_fake, True)
        self.g_loss_a2a = self.L1_lambda * self.criterion_cycle(self.real_A, self.fake_A_, self.options.padding)
        self.g_loss_b2b = self.L1_lambda * self.criterion_cycle(self.real_B, self.fake_B_, self.options.padding)
        self.g_ssim_a = self.gamma * ssim_loss(self.real_A, self.fake_B)
        self.g_ssim_b = self.gamma * ssim_loss(self.real_B, self.fake_B)
        self.g_cycle_loss = self.L1_lambda * self.criterion_cycle(self.real_A, self.fake_A_, self.options.padding) \
                            + self.L1_lambda * self.criterion_cycle(self.real_B, self.fake_B_, self.options.padding)
        self.g_disc_loss = self.D_lambda_var * self.criterionGAN(self.DA_fake, True) \
                           + self.D_lambda_var * self.criterionGAN(self.DB_fake, True)

        self.g_loss = self.D_lambda_var * self.criterionGAN(self.DA_fake, True) \
                      + self.D_lambda_var * self.criterionGAN(self.DB_fake, True) \
                      + self.L1_lambda * self.criterion_cycle(self.real_A, self.fake_A_, self.options.padding) \
                      + self.L1_lambda * self.criterion_cycle(self.real_B, self.fake_B_, self.options.padding) \
                      + self.g_ssim_a + self.g_ssim_b

        if self.id_lambda > 0.0:
            self.g_loss += self.identity_lambda * self.criterion_cycle(self.real_A, self.fake_B, self.options.padding) \
                           + self.identity_lambda * self.criterion_cycle(self.real_B, self.fake_A, self.options.padding)

        self.fake_A_sample = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32, [None, None, None, self.output_c_dim], name='fake_B_sample')

        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, True)
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, False)
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, True)
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, False)
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        self.t_sum = tf.summary.merge_all()
        self.g_loss_a2a_sum = tf.summary.scalar("g_loss_a2a", self.g_loss_a2a)
        self.g_loss_b2b_sum = tf.summary.scalar("g_loss_b2b", self.g_loss_b2b)
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_cycle_loss_sum = tf.summary.scalar("g_cycle_loss", self.g_cycle_loss)
        self.g_disc_loss_sum = tf.summary.scalar("g_disc_loss", self.g_disc_loss)
        self.g_sum = tf.summary.merge([self.g_loss_sum, self.g_disc_loss_sum, self.g_cycle_loss_sum,
                                       self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_a2a_sum,
                                       self.g_loss_b2b_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum])

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars:
            print(var.name)

    def train(self, continue_train):
        """Train cyclegan"""
        self._copy_config()
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.writer = tf.summary.FileWriter(os.path.join(self.output_dir, self.run_name, 'logs'), self.sess.graph)

        if continue_train:
            if self.load():
                print(" [*] Loading of the model was successful!")
            else:
                print(" [!] Load failed...")

        source_generator, target_generator = self._get_data_generators()
        progress_size = min(8, self.batch_size)
        self.progress_images_source, _ = source_generator.batch(progress_size)
        self.progress_images_target, _ = target_generator.batch(progress_size)
        D_lambda = self.D_lambda

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        counter = 0
        start_time = time.time()
        for epoch in range(self.epochs):
            print("Epoch: [{}], time: {:4.4f}".format(epoch, time.time() - start_time))
            dataA, _ = source_generator.batch(self.batch_size)
            dataB, _ = target_generator.batch(self.batch_size)
            target_generator.load()
            source_generator.load()
            lr = self.learning_rate * 0.5 ** (epoch // self.lr_decay_epoch)
            identity_lambda = float(max(self.id_lambda - epoch * 0.25 * self.id_lambda, 0))

            for idx in range(0, self.batch_size, self.mini_batch_size):
                batch_images = np.concatenate(
                    (dataA[self.spacing]['patches'][idx:idx + self.mini_batch_size],  # TMA-CHANGE
                     dataB[self.spacing]['patches'][idx:idx + self.mini_batch_size]), axis=3)

                fake_A, fake_B = self._gen_iteration(D_lambda, batch_images, counter, epoch, identity_lambda, idx, lr)
                self._disc_iteration(batch_images, counter, fake_A, fake_B, lr)

                if np.mod(counter, 200) == 0:
                    self._sample_progress_images(counter)
                if np.mod(counter, 50) == 0:
                    self.save(counter)
                counter += 1

            target_generator.wait()
            source_generator.wait()
            target_generator.transfer()
            source_generator.transfer()

        source_generator.stop()
        target_generator.stop()
        self.save(counter)
        print(f"training finished in: {time.time() - start_time}")

    def _disc_iteration(self, batch_images, counter, fake_A, fake_B, lr):
        disc_feed = {self.real_data: batch_images, self.fake_A_sample: fake_A, self.fake_B_sample: fake_B, self.lr: lr}
        _, summary_str = self.sess.run([self.d_optim, self.d_sum], feed_dict=disc_feed)
        self.writer.add_summary(summary_str, counter)

    def _gen_iteration(self, D_lambda, batch_images, counter, epoch, identity_lambda, idx, lr):
        gen_feed = {self.real_data: batch_images, self.lr: lr, self.identity_lambda: identity_lambda,
                    self.D_lambda_var: D_lambda}

        if epoch < self._disc_standalone_epochs: # warmup discriminator
            fake_A, fake_B = self.sess.run([self.fake_A, self.fake_B], feed_dict=gen_feed)
            return fake_A, fake_B

        if self.debug_data and np.mod(counter, 200) == 0:
            fake_A, fake_B, _, summary_str, summary_t, fake_A_, fake_B_ = self.sess.run(
                [self.fake_A, self.fake_B, self.g_optim, self.g_sum, self.t_sum, self.fake_A_, self.fake_B_],
                feed_dict=gen_feed)
            self._write_diff_images(batch_images[:, :, :, :3], fake_A_, batch_images[:, :, :, 3:], fake_B_, epoch, idx)
            self._sample_model(epoch, idx, batch_images)
            self.writer.add_summary(summary_t, counter)
        else:
            fake_A, fake_B, _, summary_str = self.sess.run([self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                                                           feed_dict=gen_feed)
        self.writer.add_summary(summary_str, counter)
        return fake_A, fake_B

    def _copy_config(self):
        config_dir = os.path.join(self.output_dir, self.run_name, 'config')
        print("creating config dir {}..".format(config_dir))
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        shutil.copy(self.param_file_path, config_dir)

    def _get_data_generators(self):
        target_generator = get_generator_from_config(config_path=self.param_file_path,
                                                     data_config_path=self.data_file_path,
                                                     generator_key='target')
        source_generator = get_generator_from_config(config_path=self.param_file_path,
                                                     data_config_path=self.data_file_path,
                                                     generator_key='source')

        print("starting generators")
        target_generator.start()
        source_generator.start()
        print("stepping generators")
        target_generator.step()
        source_generator.step()
        target_generator.wait()
        source_generator.wait()
        print("filling generators")
        source_generator.fill()
        target_generator.fill()
        source_generator.wait()
        target_generator.wait()

        return source_generator, target_generator

    def save(self, step):
        model_name = "cyclegan.model"
        checkpoint_dir = os.path.join(self.output_dir, self.run_name, 'checkpoint')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self):
        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join(self.output_dir, self.run_name, 'checkpoint')
        checkpoint_dir = checkpoint_dir.replace("\\", '/')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        print(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def _sample_progress_images(self, counter):
        print("sampling progress images..")
        for idx in range(0, self.progress_images_source[self.spacing]['patches'].shape[0], self.mini_batch_size):
            batch_images = np.concatenate((self.progress_images_source[self.spacing]['patches'][idx:idx + self.mini_batch_size],
                                           self.progress_images_target[self.spacing]['patches'][idx:idx + self.mini_batch_size]), axis = 3)
            fake_A, fake_B, fake_A_, fake_B_ = self.sess.run(
                [self.fake_A, self.fake_B, self.fake_A_, self.fake_B_],
                feed_dict={self.real_data: batch_images}
            )
            fake_A = np.concatenate([*fake_A], axis=0)[None]
            fake_B = np.concatenate([*fake_B], axis=0)[None]
            fake_A_ = np.concatenate([*fake_A_], axis=0)[None]
            fake_B_ = np.concatenate([*fake_B_], axis=0)[None]

            imgs_source = np.concatenate([np.concatenate([*batch_images[:,:,:,:3]],axis=0)[None], fake_B, fake_A_], axis=2)
            imgs_target = np.concatenate([np.concatenate([*batch_images[:,:,:,3:]],axis=0)[None], fake_A, fake_B_], axis=2)
            save_images(imgs_source, [1, 1], '{}/progress_image_source_{}_{}.jpg'.format(self.sample_dir, idx, counter), self.normalization)
            save_images(imgs_target, [1, 1], '{}/progress_image_target_{}_{}.jpg'.format(self.sample_dir, idx, counter), self.normalization)

    def _sample_model(self, epoch, idx, sample_images):
        fake_A, fake_B, fake_A_, fake_B_ = self.sess.run(
            [self.fake_A, self.fake_B, self.fake_A_, self.fake_B_],
            feed_dict={self.real_data: sample_images}
        )

        save_images(fake_A, [self.mini_batch_size, 1],
                    '{}/B_{:d}_{:d}_fakeA.jpg'.format(self.sample_dir, epoch, idx), self.normalization)
        save_images(fake_B, [self.mini_batch_size, 1],
                    '{}/A_{:d}_{:d}_fakeB.jpg'.format(self.sample_dir, epoch, idx), self.normalization)

        crop = (sample_images.shape[2] - fake_A.shape[2]) // 2
        if crop > 0:
            orig_a = sample_images[:, crop:-crop, crop:-crop, 0:3]
            orig_b = sample_images[:, crop:-crop, crop:-crop, 3:6]
        else:
            orig_a = sample_images[:, :, :, 0:3]
            orig_b = sample_images[:, :, :, 3:6]

        save_images(orig_a, [self.mini_batch_size, 1],
                    '{}/A_{:d}_{:d}_origA.jpg'.format(self.sample_dir, epoch, idx), self.normalization)
        save_images(orig_b, [self.mini_batch_size, 1],
                    '{}/B_{:d}_{:d}_origB.jpg'.format(self.sample_dir, epoch, idx), self.normalization)
        save_images(fake_A_, [self.mini_batch_size, 1],
                    '{}/A_{:d}_{:d}_cycle.jpg'.format(self.sample_dir, epoch, idx), self.normalization)
        save_images(fake_B_, [self.mini_batch_size, 1],
                    '{}/B_{:d}_{:d}_cycle.jpg'.format(self.sample_dir, epoch, idx), self.normalization)

    def _write_diff_images(self, A, fake_A_, B, fake_B_, epoch, idx):
        A_diff = (np.abs(A - fake_A_) / 2)
        B_diff = (np.abs(B - fake_B_) / 2)
        A_diff *= (A_diff > np.percentile(A_diff, q=90))
        B_diff *= (B_diff > np.percentile(B_diff, q=90))
        save_images(A_diff - 0.5, [self.mini_batch_size, 1],
                    '{}/A_{:d}_{:d}_diff.jpg'.format(self.sample_dir, epoch, idx), 0.5)
        save_images(B_diff - 0.5, [self.mini_batch_size, 1],
                    '{}/B_{:d}_{:d}_diff.jpg'.format(self.sample_dir, epoch, idx), 0.5)

    def predict(self, image, a2b=True):
        placeholder = np.zeros(image.shape)

        if a2b:
            real_data = np.concatenate((image, placeholder), axis=3)
            result = self.sess.run(self.fake_B, feed_dict={self.real_data: real_data})
        else:
            real_data = np.concatenate((placeholder, image), axis=3)
            result = self.sess.run(self.fake_A, feed_dict={self.real_data: real_data})
        return result