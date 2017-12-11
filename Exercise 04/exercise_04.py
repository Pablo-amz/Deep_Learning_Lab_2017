
# Pablo de Andres
# Deep Learning lab, ML track, exercise 4
# Uni Freiburg WS 2017/18

import argparse
import numpy as np
import os
import sys
import tempfile


from main import Data
from unet import unet

import tensorflow as tf

FLAGS = None

# For running on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tensorboard --logdir=/home/andresp/lab/tensorflow/


def main(_):

  data = Data()

  # Create the model
  x = tf.placeholder(tf.float32, [None, 300, 300, 1])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int32, [None, 116, 116])

  # Build the graph for the deep net
  y_conv = unet(x)


  with tf.name_scope('loss'):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
  loss = tf.reduce_mean(loss)
  tf.summary.scalar('loss', loss)

  with tf.name_scope('AdamOptimizer'):
    train_step = tf.train.AdamOptimizer(0.0001, 0.95, 0.99).minimize(loss)

  with tf.name_scope('Accuracy'):
    # Training accuracy
    prediction = tf.argmax(y_conv, 3) # argmax the 3rd dimension, the label
    correct_pix_pred = np.sum(prediction == y_)
    incorrect_pix_pred = np.sum(prediction != y_)
    n_pix = 116 * 116
    train_acc = correct_pix_pred / (incorrect_pix_pred + n_pix)
    # Validation accuracy
    val_acc = 0
    val_images = data.get_test_image_list_and_label_list()
    for i in range(len(val_images[0])):
      x_val = val_images[0][i]
      y_val = val_images[1][i]
      y_conv_acc = unet(x_val)
      prediction = tf.argmax(y_conv_acc, 3)
      correct_pix_pred = np.sum(prediction == y_val)
      incorrect_pix_pred = np.sum(prediction != y_val)
      n_pix = 116 * 116
      val_acc += correct_pix_pred / (incorrect_pix_pred + n_pix)

  train_acc = tf.summary.scalar('Training accuracy', train_acc)
  val_acc = tf.summary.scalar('Validation accuracy', val_acc)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  writer = tf.summary.FileWriter(graph_location)
  writer.add_graph(tf.get_default_graph())


  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(FLAGS.log_dir + '/assignment04', sess.graph)
    for i in range(400):
      batch = data.get_train_image_list_and_label_list()
      if i % 100 == 0:
        train_acc, val_acc = sess.run([train_acc, val_acc], 
                                      feed_dict={x: batch[0], y_: batch[1]})  
        writer.add_summary(train_acc, i)
        writer.add_summary(val_acc, i)
        print('step %d'% (i))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    writer.close()


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
    

