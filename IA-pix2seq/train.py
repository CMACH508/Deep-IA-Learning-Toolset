#   Copyright 2022 Sicong Zang
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""IA-pix2seq training process file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from cStringIO import StringIO
from io import StringIO
import json
import os
import time
import urllib
import zipfile
import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.linalg import block_diag

from model import Model
import utils
import sample

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
plt.switch_backend('agg')

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

# Dataset directory
tf.app.flags.DEFINE_string(
    'data_dir',
    'dataset_path',
    'The directory in which to find the dataset specified in model hparams. '
    )

# Checkpoint directory
tf.app.flags.DEFINE_string(
    'log_root', 'checkpoint_path',
    'Directory to store model checkpoints, tensorboard.')

# Resume training or not
tf.app.flags.DEFINE_boolean(
    'resume_training', True,
    'Set to true to load previous checkpoint')

# Model parameters (user defined)
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined below')


def get_default_hparams():
    """ Return default and initial HParams """
    hparams = tf.contrib.training.HParams(
        categories=['bee', 'bus', 'flower', 'giraffe', 'pig'],  # Categories used for training
        num_steps=1000001,  # Number of total steps (the process will stop automatically if the cost does not improve)
        save_every=1,  # Number of epochs before saving model
        dec_rnn_size=2048,  # Size of decoder
        dec_model='hyper',  # Decoder: lstm, layer_norm or hyper
        max_seq_len=-1,  # Max sequence length. Computed by DataLoader
        z_size=128,  # Size of latent vector z. Recommend 128.
        batch_size=200,  # Minibatch size. Recommend leaving at 128
        num_mixture=5,  # Recommend to set to the number of categories
        num_sub=2,  # Number of components for each category
        learning_rate=0.001,  # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        min_learning_rate=0.00001,  # Minimum learning rate.
        grad_clip=1.,  # Gradient clipping. Recommend leaving at 1.0.
        de_weight=0.5,  # Weight for deconv loss
        use_recurrent_dropout=True,  # Dropout with memory loss. Recomended leaving True
        recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep
        use_input_dropout=False,  # Input dropout. Recommend leaving False.
        input_dropout_prob=0.90,  # Probability of input dropout keep
        use_output_dropout=False,  # Output droput. Recommend leaving False.
        output_dropout_prob=0.9,  # Probability of output dropout keep.
        random_scale_factor=0.10,  # Random scaling data augmention proportion.
        augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
        png_scale_ratio=0.98,  # Min scaling ratio
        png_rotate_angle=0,  # Max rotating angle (abs value)
        png_translate_dist=0,  # Max translating distance (abs value)
        is_training=True,  # Training mode or not
        png_width=48,  # Width of input images
        repeat=1,  # Repeat times for EM algorithm
        semi_percent=0.0,  # Percentage of the labeled samples
        semi_balanced=False,  # Whether the labeled samples are balanced among categories
        num_per_category=70000  # How many training samples are taken from each category, [0, 70000]
    )
    return hparams


def evaluate_model(sess, model, data_set):
    """ Evaluating process """
    alpha_loss = 0.0
    gaussian_loss = 0.0
    lil_loss = 0.0
    de_loss = 0.0

    for batch in range(data_set.num_batches):
        seqs, pngs, labels, seq_len = data_set.get_batch(batch)

        feed = {
            model.global_: 1e5,
            model.input_seqs: seqs,
            model.sequence_lengths: seq_len,
            model.input_pngs: pngs
        }
        alpha_cost, gaussian_cost, lil_cost, de_cost = sess.run([model.alpha_loss, model.gaussian_loss, model.lil_loss, model.de_loss], feed)
        alpha_loss += alpha_cost
        gaussian_loss += gaussian_cost
        lil_loss += lil_cost
        de_loss += de_cost

    alpha_loss /= (data_set.num_batches)
    gaussian_loss /= (data_set.num_batches)
    lil_loss /= (data_set.num_batches)
    de_loss /= (data_set.num_batches)
    return alpha_loss, gaussian_loss, lil_loss, de_loss

def _train(sess, model, train_set, train_label_mask, epoch, sum):
    """ Training process """
    start = time.time()

    index = np.arange(len(train_set.strokes))
    np.random.shuffle(index)
    count = 0

    for begin, end in zip(range(0, len(index), model.hps.batch_size), range(model.hps.batch_size, len(index), model.hps.batch_size)):
        batch_index = index[begin:end]
        mask = train_label_mask[batch_index]
        seqs, pngs, labels, seq_len = train_set._get_batch_from_indices(batch_index)

        feed = {
            model.global_: sum,
            model.input_seqs: seqs,
            model.sequence_lengths: seq_len,
            model.input_pngs: pngs
        }
        lr, alpha_cost, gaussian_cost, lil_cost, de_cost, _, _, _, _ = \
            sess.run([model.lr, model.alpha_loss, model.gaussian_loss, model.lil_loss, model.de_loss, model.train_op,
                      model.update_gmm_mu, model.update_gmm_sigma2, model.update_gmm_alpha], feed)
        count += 1
        sum += 1

        # Record the value of losses
        if count % 20 == 0:
            f1 = open('./alpha.txt', 'a')
            f1.write(str(alpha_cost))
            f1.write('\n')
            f1.close()

            f2 = open('./L2.txt', 'a')
            f2.write(str(lil_cost))
            f2.write('\n')
            f2.close()

            f3 = open('./gau.txt', 'a')
            f3.write(str(gaussian_cost))
            f3.write('\n')
            f3.close()

            f3 = open('./de.txt', 'a')
            f3.write(str(de_cost))
            f3.write('\n')
            f3.close()

            end = time.time()
            time_taken = end - start
            start = time.time()

            print('Epoch: %d, Step: %d, Lr: %.6f, Alpha: %.2f, Gau: %.2f, Lil: %.2f, De: %.2f, Time: %.2f,'
                  % (epoch, count, lr, alpha_cost, gaussian_cost, lil_cost, de_cost, time_taken))
    epoch += 1
    return epoch, sum


def _validate(sess, eval_model, valid_set):
    """ Validating process """
    start = time.time()
    valid_alpha_loss, valid_gaussian_loss, valid_lil_loss, valid_de_loss = evaluate_model(sess, eval_model, valid_set)
    end = time.time()
    time_taken_valid = end - start
  
    print('Alpha: %.4f, Gau: %.4f, Lil: %.4f, De %.4f, Time_taken: %.4f' %
          (valid_alpha_loss, valid_gaussian_loss, valid_lil_loss, valid_de_loss, time_taken_valid))
    return valid_lil_loss
  

def _test(sess, eval_model, test_set):
    """ Testing process """
    start = time.time()
    test_alpha_loss, test_gaussian_loss, test_lil_loss, test_de_loss = evaluate_model(sess, eval_model, test_set)
    end = time.time()
    time_taken_test = end - start
  
    print('Alpha: %.4f, Gau: %.4f, Lil: %.4f, De %.4f, Time_taken: %.4f' %
          (test_alpha_loss, test_gaussian_loss, test_lil_loss, test_de_loss, time_taken_test))


def prepare(model_params):
    """ Prepare data and model for training """
    raw_data = utils.load_data(FLAGS.data_dir, model_params.categories, model_params.num_per_category)
    train_set, valid_set, test_set, max_seq_len = utils.preprocess_data(raw_data,
                                                                        model_params.batch_size,
                                                                        model_params.random_scale_factor,
                                                                        model_params.augment_stroke_prob,
                                                                        model_params.png_scale_ratio,
                                                                        model_params.png_rotate_angle,
                                                                        model_params.png_translate_dist)
    model_params.max_seq_len = max_seq_len

    # Data with labels
    index = np.arange(len(train_set.strokes))
    np.random.shuffle(index)
    index_with_label = index[0:int(len(train_set.strokes) * model_params.semi_percent)]
    train_label_mask = np.zeros([len(train_set.strokes)])
    for i in range(len(index_with_label)):
        train_label_mask[index_with_label[i]] = 1.

    # Evaluating model params
    eval_model_params = utils.copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = False
    
    # Reset computation graph and build model
    utils.reset_graph()
    train_model = Model(model_params)
    eval_model = Model(eval_model_params, reuse=True)
    
    # Create new session
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())
    
    # Load checkpoint if resume training
    if FLAGS.resume_training:
        sess, epoch, count, best_valid_cost = load_checkpoint(sess, FLAGS.log_root)
    else:
        best_valid_cost = 1e20  # set a large init value
        epoch = 1
        count = 0

    # Save model params to a json file
    tf.gfile.MakeDirs(FLAGS.log_root)
    with tf.gfile.Open(os.path.join(FLAGS.log_root, 'model_config.json'), 'w') as f:
        json.dump(model_params.values(), f, indent=True)

    return sess, train_model, eval_model, train_set, train_label_mask, valid_set, test_set, best_valid_cost, epoch, count

def load_checkpoint(sess, log_root):
    """ Load checkpoints"""
    utils.load_checkpoint(sess, log_root)
    file = np.load(FLAGS.log_root + "/para.npz")
    best_valid_cost = float(file['best_valid_loss'])
    epoch = int(file['epoch'])  # Last epoch during training
    count = int(file['count'])  # Previous accumulated steps for training
    return sess, epoch, count, best_valid_cost

def train_model(model_params):
    """ Main branch for RPCLVQ """
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
    sess, model, eval_model, train_set, train_label_mask, valid_set, test_set, best_valid_cost, epoch, count = prepare(model_params)

    cnt = 0  # Number of invalid training epoch
    for _ in range(100000):
        epoch, count = _train(sess, model, train_set, train_label_mask, epoch, count)

        if (epoch % model_params.save_every) == 0:
            print('Best_valid_loss: %4.4f' % best_valid_cost)
            valid_cost = _validate(sess, eval_model, valid_set)

            if best_valid_cost > valid_cost:
                best_valid_cost = valid_cost

                # Save model to checkpoint path
                start = time.time()
                utils.save_model(sess, FLAGS.log_root, epoch)

                np.savez("./para", best_valid_loss=best_valid_cost, epoch=epoch, count=count)
                end = time.time()
                time_taken_save = end - start
                print('time_taken_save %4.4f.' % time_taken_save)

                _test(sess, eval_model, test_set)
                cnt = 0
            else:  # Reload the last checkpoint
                sess, epoch, count, best_valid_cost = load_checkpoint(sess, FLAGS.log_root)
                cnt += 1

            if cnt >= 5:  # No improvement on validation cost for five validation steps
                print("===================================")
                print("           No Improvement          ")
                print("===================================")
                break


def main(unused_argv):
    """Load model params, save config file and start trainer."""
    model_params = get_default_hparams()
    if FLAGS.hparams:
        model_params.parse(FLAGS.hparams)
    train_model(model_params)


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
