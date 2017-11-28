
# Adaptation of the demo code by the TensorFlow authors
# Added Tensorboard functionality and tune parameters to the DL lab, excercise 3

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
"""


import argparse
import matplotlib.pyplot as plt
import os
import sys
import tempfile
import time


from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# For running on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# For opening tensorboard
# tensorboard --logdir=/home/<username>/lab/tensorflow/
learn_rate = 1e-2


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """


  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 1, 8])
    b_conv1 = bias_variable([8])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, 8, 4])
    b_conv2 = bias_variable([4])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Third convolutional layer
  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([3, 3, 4, 2])
    b_conv3 = bias_variable([2])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

  # First transposed convolution layer
  with tf.name_scope('transconv1'):
    h_trans1 = transpose_convolution(h_conv3, 4)

  # Fourth convolutional layer
  with tf.name_scope('conv4'):
    W_conv4 = weight_variable([3, 3, 4, 4])
    b_conv4 = bias_variable([4])
    h_conv4 = tf.nn.relu(conv2d(h_trans1, W_conv4) + b_conv4)

  # Second transposed convolution layer
  with tf.name_scope('transconv2'):
    h_trans2 = transpose_convolution(h_conv4, 8)

  # Fifth convolutional layer
  with tf.name_scope('conv5'):
    W_conv5 = weight_variable([3, 3, 8, 8])
    b_conv5 = bias_variable([8])
    h_conv5 = tf.nn.relu(conv2d(h_trans2, W_conv5) + b_conv5)


  with tf.name_scope('outputconv'):
    W_outputconv = weight_variable([1, 1, 8, 1])
    b_outputconv = bias_variable([1])
    h_outputconv = conv2d(h_conv5, W_outputconv) + b_outputconv

  return h_outputconv

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def transpose_convolution(inputs, filters):
  return tf.layers.conv2d_transpose(inputs, filters, [2, 2], strides=[2, 2],
                                    padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Build the graph for the deep net
  y_conv = deepnn(x)
  img_autoenc = tf.summary.image("autoencoder/rate" + str(learn_rate), y_conv, max_outputs=2)
  y_conv = tf.reshape(y_conv, [64, 784])


  with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y_conv - x))
  tf.summary.scalar('loss', loss)

  with tf.name_scope('AdamOptimizer'):
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  writer = tf.summary.FileWriter(graph_location)
  writer.add_graph(tf.get_default_graph())



  with tf.Session() as sess:
    # start_time = time.time()
    sess.run(tf.global_variables_initializer())
    sumv = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(FLAGS.log_dir + '/learningRate' 
                                   + str(learn_rate), sess.graph)
    for i in range(100000):
      # Don't need the labels anymore
      batch, _ = mnist.train.next_batch(64)
      if i % 100 == 0:
        summ, train_loss, img = sess.run([sumv, loss, img_autoenc],
                                          feed_dict={x: batch})    
        writer.add_summary(summ, i)
        writer.add_summary(img, i)
        print('step %d, training loss %g' % (i, train_loss))
      train_step.run(feed_dict={x: batch})
    writer.close()
    # end_time = time.time()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str,
                      default=os.path.join(os.getenv('HOME', '/home'),
                                           'lab/tensorflow'),
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

