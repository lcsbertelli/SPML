# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training a CNN on MNIST with Keras and the DP SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import keras as K
import numpy as np
import tensorflow.compat.v1 as tf
import math

from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
import time
from keras.models import load_model

AdamOptimizer = tf.train.AdamOptimizer

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_float('delta', 0.0002, 'Learning rate for training')
flags.DEFINE_integer('size', 3000, 'subset for training')
#flags.DEFINE_float('noise_multiplier',1.193,'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 5, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 25, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
FLAGS = flags.FLAGS


def load_cifar(noise):

    (train_images, train_labels), (test_images_,test_labels_) = datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_data = np.array(train_images[:FLAGS.size], dtype=np.float32) / 255
    train_labels = np.array(train_labels[:FLAGS.size], dtype=np.int32)

    test_data = np.array(test_images_[:FLAGS.size], dtype=np.float32) / 255
    test_labels = np.array(train_labels[:FLAGS.size], dtype=np.int32)

    return train_data, train_labels,test_data,test_labels

def training(noise):
    print("Noise Mutiplier",noise)
    if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
        raise ValueError('Number of microbatches should divide evenly batch_size')

    # Load training and test data.
    train_data, train_labels,test_data,test_labels = load_cifar(noise)
    #modelName="Cifarmodelnative.h5"
    modelName="Cifarmodel"+str(noise)+".h5"
    model = load_model("cifarhw/"+modelName)

    if FLAGS.dpsgd:
        optimizer = DPAdamGaussianOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=noise,
            num_microbatches=FLAGS.microbatches,
            learning_rate=FLAGS.learning_rate)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = K.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE)
        # print("loss=", loss)
    else:
        optimizer = AdamOptimizer(learning_rate=FLAGS.learning_rate)
        loss = K.losses.SparseCategoricalCrossentropy(from_logits=True)
        # print("loss=", loss)
    # Compile model with Keras
    start_time = time.time()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print('\n# Evaluate on test data')
    results = model.evaluate(test_data, test_labels, batch_size=FLAGS.batch_size)
    #print('test loss, test acc:', results)
    print("Accuracy:",results[1]*100)

    print("Latency --- %s seconds ---" % (time.time() - start_time))
    # Compute the privacy budget expended.
    if FLAGS.dpsgd:
        print('Trained with DP')
    else:
        print('Trained with vanilla non-private SGD optimizer')


def main(unused_argv):
  print("We are using Tensorflow version", tf.__version__)
  print("Keras API version: {}".format(K.__version__))
  #for i in [30,3.030,1.76,1.136,.914,.792]:
  for i in [30]:
      for iteration in range(1,2):
          #print("=====================Iteration==========================",iteration)
          training(i)

if __name__ == '__main__':
  app.run(main)

