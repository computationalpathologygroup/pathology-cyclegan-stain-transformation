"""
Some codes from https://github.com/Newmu/dcgan_code
"""

from digitalpathology.generator.batch.batchgenerator import BatchGenerator
from digitalpathology.generator.batch.batchsource import BatchSource
from digitalpathology.adapters.batchadapterbuilder import BatchAdapterBuilder
import yaml
import math
import pprint
import scipy.misc
import numpy as np
import copy
from PIL import Image
try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = img/127.5 - 1
    return img

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    if not is_testing:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

# -----------------------------

def save_images(images, size, image_path, normalization):
    return imsave(images, size, image_path, normalization)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path, normalization):
    merged_img = merge(images, size)
    rescaled_img = (merged_img + normalization) * (255 / (normalization*2))
    clipped_img = rescaled_img.clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(clipped_img)
    pil_img.save(path)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def crop_center(image, shape):
    img_shape = image.shape
    crop = [(img_shape[1] - shape[1]) // 2, (img_shape[2] - shape[2]) // 2]
    if crop[0] > 0:
        return image[:,crop[0]:-crop[0],crop[1]:-crop[1],:]
    else:
        return image

def get_config_from_yaml(config_path):
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.load(stream=param_file)

    return parameters['model'], parameters['system'], parameters['training'], parameters['data'], \
           parameters['augmentation']

def _constructgenerator(data_source, adapter, data_config, system_config, training_config, generator_key):
    buffer_size = training_config['iterations'][generator_key]['iteration count'] * \
                  training_config['iterations'][generator_key]['batch size']

    generator = BatchGenerator(label_dist=data_config['labels'][generator_key]['label ratios'],
                               patch_shapes=data_config['images']['patch shapes'],
                               spacing_tolerance=data_config['spacing tolerance'],
                               mask_spacing=data_config['labels'][generator_key]['mask pixel spacing'],
                               input_channels=data_config['images']['channels'],
                               dimension_order='BHWC',
                               label_mode=data_config['labels']['label mode'],
                               patch_sources=data_source.collection(purpose_id=generator_key, category_id=None,
                                                                    replace=True),
                               data_adapter=adapter,
                               category_dist=data_config['categories'],
                               strict_selection=data_config['labels'][generator_key]['strict selection'],
                               create_stats=False,
                               main_buffer_size=buffer_size,
                               buffer_chunk_size=training_config['buffer chunk size'],
                               read_buffer_size=buffer_size,
                               multi_threaded=system_config['multi-threaded'],
                               process_count=system_config[generator_key]['process count'],
                               sampler_count=system_config[generator_key]['sampler count'],
                               join_timeout=system_config['join timeout secs'],
                               response_timeout=system_config['response timeout secs'],
                               poll_timeout=system_config['poll timeout secs'],
                               name_tag=generator_key)

    return generator


def _constructdataadapter(data_config, augmentation_config, generator_key):
    data_adapter_builder = BatchAdapterBuilder()
    data_adapter_builder.setaugmenters(config=augmentation_config.get(generator_key))

    label_config = {key: data_config['labels'][key] for key in data_config['labels'] if
                    key not in ('training', 'validation')}
    label_config.update(data_config['labels'][generator_key])
    data_adapter_builder.setlabelmapper(config=label_config)

    if data_config['range normalization']['enabled']:
        data_adapter_builder.setrangenormalizer(config=data_config['range normalization'])

    if data_config['weight mapping']['enabled']:
        data_adapter_builder.setweightmapper(config=data_config['weight mapping'])

    adapter = data_adapter_builder.build()
    return adapter


def get_generator_from_config(config_path, data_config_path, generator_key=None):
    _, system_config, training_config, data_config, augmentation_config = get_config_from_yaml(config_path)

    # Load data source and initialize data: copy or check the source files.
    #
    batch_source = BatchSource(source_items=None)
    batch_source.load(file_path=data_config_path)
    if batch_source.purposes:
        # Validate the two purpose distributions.
        #
        if not batch_source.validate(purpose_distribution=data_config['purposes']):
            raise ValueError("invalid purpose distribution")
    else:
        # The data file is not distributed yet.
        #
        batch_source.distribute(purpose_distribution=data_config['purposes'])
    #
    adapter = _constructdataadapter(data_config, augmentation_config, generator_key)

    generator = _constructgenerator(data_source=batch_source,
                                    adapter=adapter,
                                    data_config=data_config,
                                    system_config=system_config,
                                    training_config=training_config,
                                    generator_key=generator_key)

    return generator