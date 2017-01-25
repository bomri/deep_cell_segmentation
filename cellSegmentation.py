import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm
import ops  # Ops is a file with operations. Currently only conv layer implementation
import re
EPS = 1e-5

class CellSegmentation(object):
    """
    Cell segmentation model class
    """
    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None, regularization_weight=None, name=None):
        """
        :param input: data set images
        :param labels: data set labels
        :param dims_in: list input image size, for example [64,64,1] (W,H,C)
        :param dims_out: list output image size, for example [64,64,1] (W,H,C)
        :param regularization_weight: L2 Norm reg weight
        :param name: model name, used for summary writer sub-names (Must be unique!)
        """
        self.input = input
        self.labels = labels
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.regularization_weight = regularization_weight
        self.base_name = name

    def model(self, train_phase):
        """
        Define the model - The network architecture
        :param train_phase: tf.bool with True for train and False for test
        """
        # Reshape the input for batchSize, dims_in[0] X dims_in[1] image, dims_in[2] channels
        x_image = tf.reshape(self.input, [-1, self.dims_in[0], self.dims_in[1], self.dims_in[2]],
                             name='x_input_reshaped')
        # Dump input image
        tf.image_summary(self.get_name('x_input'), x_image)

        # TODO - max pooling
        # TODO - init weights method

        # Model convolutions
        d_out = 1

        # conv_1, reg1 = ops.conv2d(x_image, output_dim=d_out, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_1")
        conv_1, reg1 = ops.conv2d(x_image, output_dim=16, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_1")
        conv_1 = batch_norm_layer(x=conv_1, train_phase=train_phase, scope_bn="bn1")
        conv_1 = ops.lrelu(conv_1, name='relu1')

        # conv_2, reg2 = ops.conv2d(conv_1, output_dim=d_out, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_2")
        conv_2, reg2 = ops.conv2d(conv_1, output_dim=32, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_2")
        conv_2 = batch_norm_layer(x=conv_2, train_phase=train_phase, scope_bn="bn2")
        conv_2 = ops.lrelu(conv_2, name='relu2')

        conv_3, reg3 = ops.conv2d(conv_2, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_3")
        conv_3 = batch_norm_layer(x=conv_3, train_phase=train_phase, scope_bn="bn3")
        conv_3 = ops.lrelu(conv_3, name='relu3')

        conv_4, reg4 = ops.conv2d(conv_3, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_4")
        conv_4 = batch_norm_layer(x=conv_4, train_phase=train_phase, scope_bn="bn4")
        conv_4 = ops.lrelu(conv_4, name='relu4')

        conv_5, reg5 = ops.conv2d(conv_4, output_dim=d_out, k_h=3, k_w=3, d_h=1, d_w=1, name="conv_5")

        predict = conv_5
        
        # reg = reg1 # reg2 + reg3 + reg4
        reg = reg1 + reg2 + reg3 + reg4 + reg5
        return predict, reg

    def new_model(self, train_phase):
        """
        Define the model - The network architecture
        :param train_phase: tf.bool with True for train and False for test
        """
        # Reshape the input for batchSize, dims_in[0] X dims_in[1] image, dims_in[2] channels
        x_image = tf.reshape(self.input, [-1, self.dims_in[0], self.dims_in[1], self.dims_in[2]],
                             name='x_input_reshaped')
        output_size = 1
        # Dump input image
        tf.image_summary(self.get_name('x_input'), x_image)

        # Model convolutions:

        # conv1 + relu
        with tf.variable_scope('conv1') as scope:
            ch1 = 1
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[3, 3, x_image.get_shape()[-1], ch1],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')

            biases = _variable_on_cpu('biases', [ch1], tf.constant_initializer(0.0))
            # self.example_biases = biases
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            # conv1_bn = my_batch_norm(conv1, ch, train_phase, name=scope.name + '_bn')
            # conv1_bn = batch_norm(conv1, is_training=train_phase)
            conv1_bn = conv1
            _activation_summary(conv1_bn)

        # pool1
        # pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')

        # norm1
        # norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        # conv2 + relu
        with tf.variable_scope('conv2') as scope:
            ch2 = 1
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[3, 3, ch1, ch2],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(conv1_bn, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [ch2], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            # conv2_bn = my_batch_norm(conv2, ch2, train_phase, name=scope.name + '_bn')
            # conv2_bn = batch_norm(conv2, is_training=train_phase)
            conv2_bn = conv2
            _activation_summary(conv2_bn)

        # norm2
        # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')

        # pool2
        # pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3 + relu
        with tf.variable_scope('conv3') as scope:
            ch3 = output_size
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[3, 3, ch2, ch3],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(conv2_bn, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [ch3], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            # conv3_bn = my_batch_norm(conv3, ch3, train_phase, name=scope.name + '_bn')
            # conv3_bn = batch_norm(conv3, is_training=train_phase)
            conv3_bn = conv3
            _activation_summary(conv3)

        # norm3
        # norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

        predict = conv3_bn
        reg = None

        return predict, reg

    def loss(self, predict, reg=None):
        """
        Return loss value
        :param predict: prediction from the model
        :param reg: regularization
        :return:
        """
        labels_image = tf.reshape(tf.cast(self.labels, tf.float16), [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='y_input_reshape')
        tf.image_summary(self.get_name('Labels'), labels_image)

        # Reshape to flatten tensors
        predict_reshaped = tf.contrib.layers.flatten(predict)
        labels = tf.contrib.layers.flatten(self.labels)
        
        # You need to choose loss function
        # loss = -999

        if True:
            print("using sigmoid_cross_entropy_with_logits as loss ")
            pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(predict_reshaped, labels)
            loss = tf.reduce_mean(pixel_loss)
        if False:
            print("using dice_coef_loss as loss")
            predict = tf.cast(tf.contrib.layers.flatten(predict > 0), tf.float32)
            # Calculate dice score
            intersection = tf.add(tf.reduce_sum(tf.multiply(predict, labels), keep_dims=True), EPS)
            union = tf.add(tf.add(tf.reduce_sum(predict, keep_dims=True), tf.reduce_sum(labels, keep_dims=True)), EPS)
            dice = tf.div((tf.multiply(2.0, intersection)), union)
            loss = 1 - dice

        tf.scalar_summary(self.get_name('loss without regularization'), loss)

        if reg is not None:
            tf.scalar_summary(self.get_name('regulariztion'), reg)

            # Add the regularization term to the loss.
            loss += self.regularization_weight * reg
            tf.scalar_summary(self.get_name('loss+reg'), loss)
        
        return loss


    def training(self, s_loss, learning_rate):
        """
        :param s_loss:
        :param learning_rate:
        :return:
        """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(self.get_name(s_loss.op.name), s_loss)
        
        # Here you can change to any solver you want

        # Create Adam optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(s_loss, global_step=global_step)
        return train_op

    def evaluation(self, predict, labels):
        """
        Calcualte dice score
        :param predict: predict tensor
        :param labels: labels tensor
        :return: Dice score [0,1]
        """

        # Please do not change this function

        predict = tf.cast(tf.contrib.layers.flatten(predict > 0), tf.float32)
        labels = tf.contrib.layers.flatten(self.labels)
        
        # Calculate dice score
        intersection = tf.reduce_sum(predict * labels, keep_dims=True) + EPS
        union = tf.reduce_sum(predict, keep_dims=True) + tf.reduce_sum(labels, keep_dims=True) + EPS 
        dice = (2 * intersection) / union

        # Return value and write summary
        ret = dice[0,0]
        tf.scalar_summary(self.get_name("Evaluation"), ret)
        return ret

    def get_name(self, name):
        """
        Get full name with prefix name
        """
        return "%s_%s" % (self.base_name, name)

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    # with tf.device('/cpu:0'):
    with tf.device('/gpu:0'):
        # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    ###############
    dtype = tf.float32
    ###############

    # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  TOWER_NAME = 'tower'
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  # tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x) # TODO this gave me an error but a fix could give me the histograms at the end
  # tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x)) # TODO this gave me an error but a fix could give me the histograms at the end

def batch_norm_layer(x, train_phase, scope_bn):
  bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
                        updates_collections=None,
                        is_training=True, scope=scope_bn)
  bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
                            updates_collections=None,
                            is_training=False, scope=scope_bn, reuse=True)
  bn = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
  return bn