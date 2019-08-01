from __future__ import division
import os
import time
from collections import namedtuple
import stroma.utils.generator as patch_gen
import yaml
import shutil
from cyclegan.module import *
from cyclegan.utils import *

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.output_dir = args['output_dir']
        self.run_name = args['run_name']
        self.sample_dir = os.path.join(self.output_dir, 'samples', self.run_name)
        self.debug_data = args['debug_data']

        self.discriminator = discriminator
        self.generator = generator_unet

        self.param_file_path = args['param_file_path']
        if args['data_file_path']:
            self.data_file_path = args['data_file_path']
            self.criterion_cycle = focal_abs_criterion
        else:
            self.criterion_cycle = abs_criterion
        self.configure_variables()
        if self.use_logloss:
            self.criterionGAN = sce_criterion
        else:
            self.criterionGAN = mae_criterion

        self._build_model()

        self.saver = tf.train.Saver()
        self.pool = ImagePool(50)

    def configure_variables(self):
        with open(file=self.param_file_path, mode='r') as param_file:
            parameters = yaml.load(stream=param_file)
        model_params = parameters['model']
        training_params = parameters['training']

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
        self.use_logloss = training_params['use_logloss']
        self.input_c_dim = model_params['input_c_dim']
        self.output_c_dim = model_params['output_c_dim']
        self.normalization = model_params['normalization']
        self._initial_standalone_generator = training_params['initial_standalone_generator']

        OPTIONS = namedtuple('OPTIONS', 'use_maxpool gf_dim df_dim output_c_dim unet_depth padding unet_residual regularization')
        self.options = OPTIONS._make((model_params['use_maxpool'], model_params['ngf'], model_params['ndf'],
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
        self.g_cycle_loss = self.L1_lambda * self.criterion_cycle(self.real_A, self.fake_A_, self.options.padding) \
                            + self.L1_lambda * self.criterion_cycle(self.real_B, self.fake_B_, self.options.padding)
        self.g_disc_loss = self.D_lambda_var * self.criterionGAN(self.DA_fake, True) \
                           + self.D_lambda_var * self.criterionGAN(self.DB_fake, True)
        self.g_loss = self.D_lambda_var * self.criterionGAN(self.DA_fake, True) \
                      + self.D_lambda_var * self.criterionGAN(self.DB_fake, True) \
                      + self.L1_lambda * self.criterion_cycle(self.real_A, self.fake_A_, self.options.padding) \
                      + self.L1_lambda * self.criterion_cycle(self.real_B, self.fake_B_, self.options.padding) \
                      + tf.losses.get_regularization_loss()
        if self.id_lambda > 0.0:
            self.g_loss += self.identity_lambda * self.criterion_cycle(self.real_A, self.fake_B, self.options.padding) \
                           + self.identity_lambda * self.criterion_cycle(self.real_B, self.fake_A, self.options.padding)

        # + self.l2_lambda * tf.nn.l2_loss(self.g_vars)
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
        self.d_loss = self.da_loss + self.db_loss  # + self.l2_lambda * tf.nn.l2_loss(self.d_vars)

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
        for var in t_vars: print(var.name)

    def train(self, continue_train):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        tvars = tf.trainable_variables()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(os.path.join(self.output_dir, 'logs', self.run_name), self.sess.graph)

        counter = 0
        start_time = time.time()

        if continue_train:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # init data generators
        config_dir = os.path.join(self.output_dir, 'config', self.run_name)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        shutil.copy(self.param_file_path, config_dir)
        source_generator, target_generator = self.get_data_generators()
        
        for epoch in range(self.epochs):
            # dataA = glob('./datasets/{}/* .*'.format(self.dataset_dir + '/trainA'))
            # dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            # np.random.shuffle(dataA)
            # np.random.shuffle(dataB)
            # dataA = next(source_generator) # TMA-CHANGE
            dataA, _ = source_generator.batch(self.batch_size)
            dataB, _ = target_generator.batch(self.batch_size)
            # batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = self.learning_rate if epoch < self.lr_decay_epoch else self.learning_rate * (self.epochs - epoch) / (
                        self.epochs - self.lr_decay_epoch)
            identity_lambda = float(max(self.id_lambda - epoch * 0.25 * self.id_lambda, 0))
            print(identity_lambda)
            if self._initial_standalone_generator:
                D_lambda = 0 if identity_lambda > 0 else self.D_lambda
            else:
                D_lambda = self.D_lambda
            target_generator.load()
            source_generator.load()  # TMA-CHANGE

            for idx in range(0, self.batch_size, self.mini_batch_size):
                # batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                #                        dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                # batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in
                #                 batch_files]
                # batch_images = np.array(batch_images).astype(np.float32)
                batch_images = np.concatenate(
                    (dataA[0]['patches'][idx:idx + self.mini_batch_size].transpose(0, 2, 3, 1),  # TMA-CHANGE
                     dataB[0]['patches'][idx:idx + self.mini_batch_size].transpose(0, 2, 3, 1)), axis=3)
                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str, summary_t, fake_A_, fake_B_, g_loss = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum, self.t_sum, self.fake_A_, self.fake_B_, self.g_loss],
                    feed_dict={self.real_data: batch_images, self.lr: lr, self.identity_lambda: identity_lambda,
                               self.D_lambda_var: D_lambda})

                if self.debug_data:
                    print("gen:")
                    tvars_vals = self.sess.run(tvars)
                    print(g_loss)
                    print(np.max([np.max(x) for x in tvars_vals]))
                    print(np.min([np.min(x) for x in tvars_vals]))
                    if np.mod(counter, 200) == 0:
                        self.write_debugger(batch_images[:,:,:,:3], fake_A_, batch_images[:,:,:,3:], fake_B_,
                                            epoch, idx)

                if np.mod(counter, 200) == 0:
                    self.writer.add_summary(summary_t, counter)
                self.writer.add_summary(summary_str, counter)
                # [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                # batch_images_ = crop_center(batch_images, fake_A.shape)

                _, summary_str, d_loss = self.sess.run(
                    [self.d_optim, self.d_sum, self.d_loss],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                if self.debug_data:
                    print("disc:")
                    print(d_loss)
                    tvars_vals = self.sess.run(tvars)
                    print(np.max([np.max(x) for x in tvars_vals]))
                    print(np.min([np.min(x) for x in tvars_vals]))
                self.writer.add_summary(summary_str, counter)


                if np.mod(counter, 200) == 0:
                    print("Epoch: [{}] [{}/{}] time: {:4.4f}".format(epoch, idx, self.batch_size,
                                                                     time.time() - start_time))
                if np.mod(counter, 200) == 0:
                    print("writing samples")
                    self.sample_model(epoch, idx, batch_images)


                if np.mod(counter, 50) == 0:
                    self.save(counter)
                counter += 1
            target_generator.wait()
            source_generator.wait()  # TMA-CHANGE
            target_generator.transfer()
            source_generator.transfer()  # TMA-CHANGE


    def get_data_generators(self):
        target_generator = patch_gen.get_generator_from_config(config_path=self.param_file_path,
                                                               data_config_path=self.data_file_path,
                                                               generator_key='target')
        source_generator = patch_gen.get_generator_from_config(config_path=self.param_file_path,
                                                               data_config_path=self.data_file_path,
                                                               generator_key='source')

        print("starting")
        target_generator.start()
        source_generator.start()
        print("stepping")
        target_generator.step()
        source_generator.step()
        target_generator.wait()
        source_generator.wait()
        print("filling")
        source_generator.fill()
        target_generator.fill()
        source_generator.wait()
        target_generator.wait()

        return source_generator, target_generator

    def save(self, step):
        model_name = "cyclegan.model"
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoint', self.run_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self):
        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoint', self.run_name)
        checkpoint_dir = checkpoint_dir.replace("\\", '/')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        print(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, epoch, idx, sample_images):
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
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

    def predict(self, image, a2b=True):
        placeholder = np.zeros(image.shape)

        if a2b:
            real_data = np.concatenate((image, placeholder), axis=3)
            result = self.sess.run(self.fake_B, feed_dict={self.real_data: real_data})
        else:
            real_data = np.concatenate((placeholder, image), axis=3)
            result = self.sess.run(self.fake_A, feed_dict={self.real_data: real_data})
        return result

    def write_debugger(self, A, fake_A_, B, fake_B_, epoch, idx):
        A_diff = (np.abs(A - fake_A_) / 2)
        B_diff = (np.abs(B - fake_B_) / 2)
        A_diff *= (A_diff > np.percentile(A_diff, q=90))
        B_diff *= (B_diff > np.percentile(B_diff, q=90))
        save_images(A_diff  - 0.5, [self.mini_batch_size, 1],
                    '{}/A_{:d}_{:d}_diff.jpg'.format(self.sample_dir, epoch, idx), 0.5)
        save_images(B_diff  - 0.5, [self.mini_batch_size, 1],
                    '{}/B_{:d}_{:d}_diff.jpg'.format(self.sample_dir, epoch, idx), 0.5)
