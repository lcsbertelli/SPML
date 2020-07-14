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
from absl import logging
import keras as K
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import math

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
import time
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from art.estimators.classification import KerasClassifier
from art.attacks.inference import MIFace
GradientDescentOptimizer = tf.train.GradientDescentOptimizer

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.25, 'Learning rate for training')
flags.DEFINE_float('delta', 0.0001, 'Learning rate for training')
flags.DEFINE_integer('size', 60000, 'subset for training')
#flags.DEFINE_float('noise_multiplier',1.193,'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 500, 'Batch size')
flags.DEFINE_integer('epochs', 50, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 20, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS

def load_mnist(noise):
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = K.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data[:FLAGS.size], dtype=np.float32) / 255
  test_data = np.array(test_data[:FLAGS.size], dtype=np.float32) / 255
  train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)

  test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

  train_labels = np.array(train_labels[:FLAGS.size], dtype=np.int32)
  test_labels = np.array(test_labels[:FLAGS.size], dtype=np.int32)

  train_labels = K.utils.to_categorical(train_labels, num_classes=10)
  test_labels = K.utils.to_categorical(test_labels, num_classes=10)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels,test_data,test_labels

def training(noise):
    print("Noise Mutiplier",noise)
    logging.set_verbosity(logging.INFO)
    if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
        raise ValueError('Number of microbatches should divide evenly batch_size')

    # Load training and test data.
    train_data, train_labels, test_data, test_labels = load_mnist(noise)
    # print("in main",train_data)
    #modelName = "Mnistmodel"+str(noise)+".h5"
    modelName = "Mnistmodel"+str(noise)+".h5"
    model = load_model("mnistfullhw/"+modelName)
    start_time = time.time()

    if FLAGS.dpsgd:
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=noise,
            num_microbatches=FLAGS.microbatches,
            learning_rate=FLAGS.learning_rate)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = K.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE)
        print("loss=", loss)
    else:
        optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        loss = K.losses.CategoricalCrossentropy(from_logits=True, )
        # print("loss=", loss)
    # Compile model with Keras
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=True)
    attack = MIFace(classifier, max_iter=10000, threshold=1.)
    y = np.arange(10)
    x_init_average = np.zeros((10, 28, 28, 1)) + np.mean(test_data, axis=0)
    x_infer_from_average = attack.infer(x_init_average, y)
    plt.figure(figsize=(15, 15))
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.imshow((np.reshape(x_infer_from_average[0 + i,], (28, 28))), cmap=plt.cm.gray_r)

    plt.savefig("Mnistattack"+str(noise)+".pdf")
    plt.show()
    print("Latency --- %s seconds ---" % (time.time() - start_time))
    if FLAGS.dpsgd:
        print('trained with DP')
    else:
        print('Trained with vanilla non-private SGD optimizer')

def main(unused_argv):
  print("We are using Tensorflow version", tf.__version__)
  print("Keras API version: {}".format(K.__version__))
  #for i in [30,3.5,1.90,1.15,.920,.810]:
  for i in [.810]:
    for iteration in range(1,2):
        print("=====================Iteration==========================",iteration)
        training(i)

if __name__ == '__main__':
  app.run(main)
