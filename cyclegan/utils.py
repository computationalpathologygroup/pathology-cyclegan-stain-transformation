"""
Some codes from https://github.com/Newmu/dcgan_code
"""

import yaml
from functools import partial

import tensorflow as tf
import numpy as np
import albumentations
import digitalpathology.generator.batch.simplesampler as sampler

def get_config_from_yaml(config_path):
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.load(stream=param_file)
    return parameters['model'], parameters['sampler'], parameters['training']

def get_generator_from_config(sampler_param, data_config_path, albumentations_path, batch_size):
    transforms = albumentations.load(albumentations_path, data_format='yaml')
    try:
        source_sampler = sampler.SimpleSampler(patch_source_filepath=data_config_path,
                                               **sampler_param['training'],
                                               partition='source')
        target_sampler = sampler.SimpleSampler(patch_source_filepath=data_config_path,
                                               **sampler_param['training'],
                                               partition='target')
    except:
        NotImplementedError("The original implementation uses an internal patch sampling library "
                            "that is not publically available")

    generator = TFDataGenerator(source_sampler, target_sampler, transforms,
                                batch_size=batch_size)
    return generator

class TFDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, source, target, augmentations_pipeline=None, batch_size=8):
        self._source = source
        self._target = target
        self._source.step()
        self._target.step()
        if augmentations_pipeline:
            self._aug_fn = partial(self.augment_fn, transform=augmentations_pipeline)
        else:
            self._aug_fn = None
        self.batch_size = batch_size


    def __getitem__(self, index):
        source, target = self._preprocess_batch(index)
        return source, target

    def __len__(self):
        return self._source._iterations // self.batch_size

    def augment_fn(self, patch, transform):
        transformed = transform(image=patch)
        return transformed["image"]

    def on_epoch_end(self):
        self._source.reset_sampler_indices()
        self._target.reset_sampler_indices()

    def _preprocess_batch(self, index):
        source = np.zeros((self.batch_size, self._source.shape[0], self._source.shape[1], 3), dtype=np.float32)
        target = np.zeros((self.batch_size, self._source.shape[0], self._source.shape[1], 3), dtype=np.float32)

        patch_ind = index * self.batch_size
        for ind, i in enumerate(range(patch_ind, patch_ind + self.batch_size)):
            patch, _, _ = self._source[i]
            patch_t, _, _= self._target[i]
            if self._aug_fn:
                patch = self._aug_fn(patch)
                patch_t= self._aug_fn(patch_t)
            source[ind] = patch / 127.5 - 1
            target[ind] = patch_t / 127.5 - 1
        return source, target



