import tensorflow as tf

def unet(x):
  """unet builds the graph for a u net for classifying images.
  Args:
    x: an input tensor with the dimensions (300, 300)
  Returns:
    A tensor of shape (1, 2), with values equal to the logits of 
    classifying the digit into one of 2 classes (cell or background).
  """


  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 300, 300, 1])
    x_image = tf.cast(x_image, tf.float32)

  # First level, first convolutional layer
  with tf.name_scope('conv1.1_down'):
    W_conv1_1 = weight_variable([3, 3, 1, 32])
    b_conv1_1 = bias_variable([32])
    h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1)

  # First level, second convolutional layer
  with tf.name_scope('conv1.2_down'):
    W_conv1_2 = weight_variable([3, 3, 32, 32])
    b_conv1_2 = bias_variable([32])
    h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1_2)

  # ---------------------------- 222222222222222222 --------------------------

  # Second level, first convolutional layer
  with tf.name_scope('conv2.1_down'):
    W_conv2_1 = weight_variable([3, 3, 32, 64])
    b_conv2_1 = bias_variable([64])
    h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1)

  # Second level, second convolutional layer
  with tf.name_scope('conv2.2_down'):
    W_conv2_2 = weight_variable( [3, 3, 64, 64])
    b_conv2_2 = bias_variable([64])
    h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2_2)

 # ---------------------------- 3333333333333333333 --------------------------

  # Third level, first convolutional layer
  with tf.name_scope('conv3.1_down'):
    W_conv3_1 = weight_variable([3, 3, 64, 128])
    b_conv3_1 = bias_variable([128])
    h_conv3_1 = tf.nn.relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1)

  # Third level, second convolutional layer
  with tf.name_scope('conv3.2_down'):
    W_conv3_2 = weight_variable([3, 3, 128, 128])
    b_conv3_2 = bias_variable([128])
    h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, W_conv3_2) + b_conv3_2)

  # Third pooling layer.
  with tf.name_scope('pool3'):
    h_pool3 = max_pool_2x2(h_conv3_2)

 # ---------------------------- 4444444444444444444 --------------------------

  # Fourth level, first convolutional layer
  with tf.name_scope('conv4.1_down'):
    W_conv4_1 = weight_variable([3, 3, 128, 256])
    b_conv4_1 = bias_variable([256])
    h_conv4_1 = tf.nn.relu(conv2d(h_pool3, W_conv4_1) + b_conv4_1)

  # Fourth level, second convolutional layer
  with tf.name_scope('conv4.2_down'):
    W_conv4_2 = weight_variable([3, 3, 256, 256])
    b_conv4_2 = bias_variable([256])
    h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, W_conv4_2) + b_conv4_2)

  # Fourth pooling layer.
  with tf.name_scope('pool4'):
    h_pool4 = max_pool_2x2(h_conv4_2)

 # ---------------------------- 5555555555555555555 --------------------------

  # Fifth level, first convolutional layer
  with tf.name_scope('conv5.1'):
    W_conv5_1 = weight_variable( [3, 3, 256, 256])
    b_conv5_1 = bias_variable([256])
    h_conv5_1 = tf.nn.relu(conv2d(h_pool4, W_conv5_1) + b_conv5_1)

  # Fifth level, second convolutional layer
  with tf.name_scope('conv5.2'):
    W_conv5_2 = weight_variable([3, 3, 256, 256])
    b_conv5_2 = bias_variable([256])
    h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1, W_conv5_2) + b_conv5_2)

  # First transposed convolution layer
  with tf.name_scope('transconv1'):
    h_trans1 = transpose_convolution(h_conv5_2, 256)

  # ---------------------------- 444444444444444444 --------------------------

  # Fourth level, third convolutional layer
  with tf.name_scope('conv4.1_up'):
    input4_1 = reuse_features(h_trans1, h_conv4_2)
    W_conv4_1 = weight_variable([3, 3, 512, 256])
    b_conv4_1 = bias_variable([256])
    h_conv4_1 = tf.nn.relu(conv2d(input4_1, W_conv4_1) + b_conv4_1)

  # Fourth level, fourth convolutional layer
  with tf.name_scope('conv4.2_up'):
    W_conv4_2 = weight_variable([3, 3, 256, 256])
    b_conv4_2 = bias_variable([256])
    h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, W_conv4_2) + b_conv4_2)

  # Second transposed convolution layer
  with tf.name_scope('transconv2'):
    h_trans2 = transpose_convolution(h_conv4_2, 128)

 # ---------------------------- 3333333333333333333 --------------------------

  # Third level, third convolutional layer
  with tf.name_scope('conv3.1_up'):
    input3_1 = reuse_features(h_trans2, h_conv3_2)
    W_conv3_1 = weight_variable([3, 3, 256, 128])
    b_conv3_1 = bias_variable([128])
    h_conv3_1 = tf.nn.relu(conv2d(input3_1, W_conv3_1) + b_conv3_1)

  # Third level, fourth convolutional layer
  with tf.name_scope('conv3.2_up'):
    W_conv3_2 = weight_variable([3, 3, 128, 128])
    b_conv3_2 = bias_variable([128])
    h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, W_conv3_2) + b_conv3_2)

  # Third transposed convolution layer
  with tf.name_scope('transconv3'):
    h_trans3 = transpose_convolution(h_conv3_2, 64)

  # ---------------------------- 222222222222222222 --------------------------

  # Second level, third convolutional layer
  with tf.name_scope('conv2.1_up'):
    input2_1 = reuse_features(h_trans3, h_conv2_2)
    W_conv2_1 = weight_variable([3, 3, 128, 64])
    b_conv2_1 = bias_variable([64])
    h_conv2_1 = tf.nn.relu(conv2d(input2_1, W_conv2_1) + b_conv2_1)

  # Second level, second convolutional layer
  with tf.name_scope('conv2.2_up'):
    W_conv2_2 = weight_variable([3, 3, 64, 64])
    b_conv2_2 = bias_variable([64])
    h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)

  # Fourth transposed convolution layer
  with tf.name_scope('transconv4'):
    h_trans4 = transpose_convolution(h_conv2_2, 32)

  # ---------------------------- 111111111111111111 --------------------------

  # First level, third convolutional layer
  with tf.name_scope('conv1.1_up'):
    input3_1 = reuse_features(h_trans4, h_conv1_2)
    W_conv1_1 = weight_variable([3, 3, 64, 32])
    b_conv1_1 = bias_variable([32])
    h_conv1_1 = tf.nn.relu(conv2d(input3_1, W_conv1_1) + b_conv1_1)

  # First level, fourth convolutional layer
  with tf.name_scope('conv1.2_up'):
    W_conv1_2 = weight_variable([3, 3, 32, 32])
    b_conv1_2 = bias_variable([32])
    h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)

  with tf.name_scope('outputconv'):
    W_outputconv = weight_variable([1, 1, 32, 2])
    b_outputconv = bias_variable([2])
    h_outputconv = conv2d(h_conv1_2, W_outputconv) + b_outputconv

  return h_outputconv

  # --------------------------------------------------------------------------

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

def transpose_convolution(inputs, filters):
  return tf.layers.conv2d_transpose(inputs, filters, [2, 2], strides=[2, 2],
                                    padding='VALID')
  # --------------------------------------------------------------------------

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.zeros(shape)
  return tf.Variable(initial)


  # --------------------------------------------------------------------------

def reuse_features(output_previous, reuse):
  """reshapes and concatenates the input with the features to reuse"""
  target = output_previous.get_shape().as_list()
  new_input = tf.concat([tf.image.resize_image_with_crop_or_pad(reuse, target[1], target[2]),
                        output_previous], 3)
  return new_input

