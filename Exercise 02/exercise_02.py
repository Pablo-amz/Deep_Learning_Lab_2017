
# Adaptation of the demo code by the TensorFlow authors
# Added Tensorboard functionality and tune parameters to the DL lab, exercise 2

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
import os
import sys
import tempfile
import time


from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# For running on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tensorboard --logdir=/home/andresp/lab/tensorflow/
learn_rate = 1e-3
n_filters = 64


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
    W_conv1 = weight_variable([3, 3, 1, n_filters])
    b_conv1 = bias_variable([n_filters])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, n_filters, n_filters])
    b_conv2 = bias_variable([n_filters])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * n_filters, 128])
    b_fc1 = bias_variable([128])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * n_filters])
    h_fc1 = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([128, 10])
    b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
  return y_conv


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('GradientDescent'):
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
    val_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  tf.summary.scalar('accuracy', accuracy)
  
  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    sumv = tf.summary.scalar('valid_acc', accuracy)
    valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/validation' + 
                                         str(n_filters), sess.graph)
    for i in range(7000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        summary, vali_accuracy = sess.run([sumv, accuracy], 
          feed_dict={x:mnist.validation.images, y_:mnist.validation.labels})
        valid_writer.add_summary(summary,i)
        print('step %d, training accuracy %g, validation accuracy %g' % (i, 
              train_accuracy, vali_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    valid_writer.close()
    end_time = time.time()
    print('Number of filters: %g, time: %g seconds.' % (n_filters, 
          end_time - start_time))
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))

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

