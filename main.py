import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from CycleGAN_tensorflow.model import cyclegan

def collect_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--run_name', dest='run_name', default='train', help='name of the current run')
    parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
    parser.add_argument('--output_dir', dest='output_dir', default=None, help='models are saved here')
    parser.add_argument('--param_file_path', dest='param_file_path', default=None, help='set the parameters file path')
    parser.add_argument('--data_file_path', dest='data_file_path', default=None, help='set the data file path')
    parser.add_argument('--debug_data', action='store_true', default=False, help='')
    args = parser.parse_args()
    return args

def main(_):
    args = vars(collect_arguments())
    if not os.path.exists(os.path.join(args['output_dir'], 'logs')):
        os.makedirs(os.path.join(args['output_dir'], 'logs'))
    if not os.path.exists(os.path.join(args['output_dir'], 'samples')):
        os.makedirs(os.path.join(args['output_dir'], 'samples'))


    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        model.train(args['continue_train'])
if __name__ == '__main__':
    tf.app.run()