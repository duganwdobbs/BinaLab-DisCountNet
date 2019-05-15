import tensorflow as tf
import Helpers.util as util

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('conv_scope',0,'Incrementer for convolutional scopes')
flags.DEFINE_integer('bn_scope'  ,0,'Incrementer for batch norm scopes')

def init_scope_vars():
  FLAGS.conv_scope = 0
  FLAGS.bn_scope = 0

def delist(net):
  if type(net) is list:
    net = tf.concat(net,-1,name = 'cat')
  return net

def lrelu(x):
  return tf.nn.leaky_relu(x)

def relu(x):
  return tf.nn.relu(x)

def conv2d(net, filters, kernel = 3, stride = 1, dilation_rate = 1, activation = None, padding = 'SAME', trainable = True, name = None, reuse = None):
  return tf.layers.conv2d(delist(net),filters,kernel,stride,padding,dilation_rate = dilation_rate, activation = activation,trainable = trainable, name = name, reuse = reuse)

def avg_pool(net, kernel = 3, stride = 1, padding = 'SAME', name = None):
  return tf.layers.average_pooling2d(net,kernel,stride,padding=padding,name=name)

def max_pool(net, kernel = 3, stride = 3, padding = 'SAME', name = None):
  return tf.layers.max_pooling2d(net,kernel,stride,padding=padding,name=name)

def conv2d_trans(net, features, kernel, stride, activation = lrelu,padding = 'SAME', trainable = True, name = None):
  return tf.layers.conv2d_transpose(net,features,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name)

def deconv(net, features = 3, kernel = 3, stride = 2, activation = None,padding = 'SAME', trainable = True, name = None):
  return tf.layers.conv2d_transpose(net,features,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name,use_bias = False)

def batch_norm(net,training=True,trainable=True,activation = None):
  with tf.variable_scope('Batch_Norm_%d'%(FLAGS.bn_scope)):
    FLAGS.bn_scope = FLAGS.bn_scope + 1
    net = tf.layers.batch_normalization(delist(net),training = training, trainable = trainable)

    if activation is not None:
      net = activation(net)

    return net

def deconvxy(net,features = None,stride = 2, activation = None,padding = 'SAME', trainable = True, name = 'Deconv_xy'):
  with tf.variable_scope(name) as scope:

    net = delist(net)

    kernel  = 3

    if features is None:
      features = net.shape[-1].value // stride

    netx = deconv(net , features  , kernel = kernel, stride = (stride,1), name = "x",  trainable = trainable)
    netx = deconv(netx, features  , kernel = kernel, stride = (1,stride), name = "xy", trainable = trainable)


    nety = deconv(net , features  , kernel = kernel, stride = (1,stride), name = "y",  trainable = trainable)
    nety = deconv(nety, features  , kernel = kernel, stride = (stride,1), name = "yx", trainable = trainable)

    netxy= deconv(net , features  , kernel = kernel, stride = stride    , name = 'xyz', trainable = trainable)

    net  = tf.concat((netx,nety,netxy),-1)
    return net

def dense_block(net,training, filters = 2, kernel = 3, kmap = 5, stride = 1,
            activation = tf.nn.relu, padding = 'SAME', trainable = True,
            name = 'Dense_Block', prestride_return = True,use_max_pool = True):
  with tf.variable_scope(name) as scope:

    net = delist(net)

    for n in range(kmap):
      out = conv2d(net,filters=filters,kernel=kernel,activation=None,padding=padding,trainable=trainable,name = '_map_%d'%n)
      net = tf.concat([net,out],-1,name = '%d_concat'%n)

    net = batch_norm(net,training,trainable,activation)
    net = tf.layers.dropout(net, .7)

    if stride is not 1:
      prestride = net
      if use_max_pool:
        net = max_pool(net,stride,stride)
      else:
        net = avg_pool(net,stride,stride)
      if prestride_return:
        return prestride, net
    return net

def atrous_block(net,filters = 8,kernel = 3,dilation = 1,kmap = 2,activation = lrelu,trainable = True,name = 'Atrous_Block'):
  newnet = []
  with tf.variable_scope(name) as scope:
    for x in range(dilation,kmap * dilation,dilation):
      # Reuse and not trainable if beyond the first layer.
      re = True  if x > dilation else None
      tr = False if x > dilation else trainable

      with tf.variable_scope("ATROUS",reuse = tf.AUTO_REUSE) as scope:
        # Total Kernel visual size: Kernel + ((Kernel - 1) * (Dilation - 1))
        # At kernel = 9 with dilation = 2; 9 + 8 * 1, 17 px
        layer = conv2d(net,filters = filters, kernel = kernel, dilation_rate = x,reuse = re,trainable = tr)
        newnet.append(layer)

    net = delist(newnet)
    return net


# Defines a function to output the histogram of trainable variables into TensorBoard
def hist_summ():
  for var in tf.trainable_variables():
    tf.summary.histogram(var.name,var)

def cmat(labels_flat,logits_flat):
  with tf.variable_scope("Confusion_Matrix") as scope:
    label_1d  = tf.reshape(labels_flat, (FLAGS.batch_size, FLAGS.imgW * FLAGS.imgH))
    logit_1d = tf.reshape(logits_flat, (FLAGS.batch_size, FLAGS.imgW * FLAGS.imgH))
    cmat_sum = tf.zeros((FLAGS.num_classes,FLAGS.num_classes),tf.int32)
    for i in range(FLAGS.batch_size):
      cmat = tf.confusion_matrix(labels = label_1d[i], predictions = logit_1d[i], num_classes = FLAGS.num_classes)
      cmat_sum = tf.add(cmat,cmat_sum)
    return cmat_sum

def l2reg(loss):
  if FLAGS.l2_loss:
    with tf.variable_scope("L2_Loss") as scope:
      l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name])
      l2 = tf.scalar_mul(.0002,l2)
      tf.summary.scalar('L2_Loss',loss)
      loss = tf.add(loss,l2)
      tf.summary.scalar('Total_Loss',loss)
  return loss

# Function to compute Mean Square Error loss
def mse_loss(labels,logits):
  with tf.variable_scope('Mean_Square_Error') as scope:
    loss = tf.losses.mean_squared_error(labels,logits)
    tf.summary.scalar('MSE_Loss',loss)
    loss = l2reg(loss)
    return loss

# A log loss for using single class heat map
def log_loss(labels,logits):
  with tf.variable_scope('Log_Loss') as scope:
    loss = tf.losses.log_loss(labels,logits)
    tf.summary.scalar('Log_Loss',loss)
    loss = l2reg(loss)
    return loss

# Loss function for tape, using cross entropy
def xentropy_loss(labels,logits):
  with tf.variable_scope("XEnt_Loss") as scope:
    labels = tf.cast(labels,tf.int32)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels,logits = logits)
    loss = tf.reduce_mean(loss)

    tf.summary.scalar('XEnt_Loss',loss)
    loss = l2reg(loss)
    return loss

# Loss function for tape, using cross entropy
def weighted_xentropy_loss(labels,logits,weight):
  with tf.variable_scope("XEnt_Loss") as scope:
    thresh   = util.threshold(labels)
    weight_h = tf.cast(thresh,tf.float32)   * tf.cast(  weight,tf.float32)
    weight_l = tf.cast(1-thresh,tf.float32) * tf.cast(1/weight,tf.float32)

    weights = weight_h + weight_l

    labels = tf.one_hot(tf.cast(thresh,tf.int32),2)

    labels = tf.squeeze(labels)
    logits = tf.squeeze(logits)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels,
                                                      logits = logits )

    loss = tf.squeeze(tf.cast(loss,tf.float32)) * tf.squeeze(tf.cast(weights,tf.float32))
    loss = tf.reduce_mean(loss)

    tf.summary.scalar('XEnt_Loss',loss)
    loss = l2reg(loss)
    return loss

def l2loss(labels,logits,weight):
  with tf.variable_scope("L2Loss") as scope:
    labels = labels / FLAGS.gt_max
    logits = logits / FLAGS.gt_max

    diff = labels - logits
    diff = diff ** 2 / 2
    loss = tf.reduce_sum(diff)
    l2   = tf.reduce_sum(diff,(1,2,3))

    tf.summary.scalar('L2Loss',loss)
    loss = l2reg(loss)
    return loss, l2

# Loss function for tape, using cross entropy
def binary_weighted_xentropy_loss(labels,logits,weight):
  with tf.variable_scope("XEnt_Loss") as scope:
    thresh   = util.threshold(labels)
    weight_h = tf.cast(thresh,tf.float32)   * tf.cast(  weight,tf.float32)
    weight_l = tf.cast(1-thresh,tf.float32) * tf.cast(1/weight,tf.float32)

    weights = weight_h + weight_l

    shape  = [-1,FLAGS.patch_size,FLAGS.patch_size,1]
    labels = tf.reshape(labels,shape)
    logits = tf.reshape(logits,shape)

    eps = 1e-3

    pos_loss = labels * tf.log(logits + eps)
    neg_loss = (1 - labels) * tf.log(1 - logits + eps)
    loss = pos_loss + neg_loss

    loss = tf.squeeze(tf.cast(loss,tf.float32)) * tf.squeeze(tf.cast(weights,tf.float32))
    loss = -tf.reduce_mean(loss)

    tf.summary.scalar('XEnt_Loss',loss)
    loss = l2reg(loss)
    return loss

# Loss function for tape, using cross entropy
def seg_loss(labels,logits):
  with tf.variable_scope("XEnt_Loss") as scope:
    labels = tf.cast(labels,tf.int32)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels,logits = logits)
    return loss

# Absolute accuracy calculation for counting
def accuracy(labels,logits):
  with tf.variable_scope("Accuracy") as scope:
    accuracy = tf.metrics.accuracy(labels = labels, predictions = logits)
    acc,up = accuracy
    tf.summary.scalar('Accuracy',tf.multiply(acc,100))
    return accuracy

def miou(labels,logits,classes):
  with tf.variable_scope("MIOU") as scope:
    miou      = tf.metrics.mean_iou(labels = labels, predictions = logits, num_classes = classes)
    _miou,op  = miou
    tf.summary.scalar('MIOU',_miou)
    return miou

def aoc_roc(labels,logits):
  with tf.variable_scope("ROC") as scope:
    roc      = tf.metrics.auc(labels = labels, predictions = logits, curve = 'ROC')
    _roc,op  = roc
    tf.summary.scalar('ROC',_roc)
    return roc

def aoc_pr(labels,logits):
  with tf.variable_scope("PR") as scope:
    pr      = tf.metrics.auc(labels = labels, predictions = logits, curve = 'PR')
    _pr,op  = pr
    tf.summary.scalar('PR',_pr)
    return pr
