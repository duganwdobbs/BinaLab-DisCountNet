import tensorflow as tf

import numpy as np

from Helpers import ops,util,generator

from Helpers.util     import Image_To_Patch,Patch_To_Image,ImSizeToPatSize,disc_label_gen
from Helpers.ops      import init_scope_vars

# Import Flag globals
flags = tf.app.flags
FLAGS = flags.FLAGS

# Segmentation Network Class File

# A network has:
# Session / Graph
# Inputs
# Inferece
# Logits
# Metrics
# Trainer
# Saver
# Runner
# TODO: MORE SUMMARIES
# SETUP DIRECTORIES
# TODO: SAVE IMAGES

class DiscNet:
  def __init__(self,training,restore,timestr,split = None):
    with tf.Graph().as_default():
      self.net_name    = "PropNet"
      print("\rSETTING UP %s NETWORK"%self.net_name,end='')
      config = tf.ConfigProto(allow_soft_placement = True)
      config.gpu_options.allow_growth = True
      self.sess        = tf.Session(config = config)

      # Setup our Data Generator
      if split is None:
        split          = 'TRAIN' if training else 'TEST'
      self.generator   = generator.DataGenerator(split,FLAGS.base_dir)

      self.global_step = tf.Variable(0,name='global_step',trainable = False)
      self.training    = training
      self.restore     = restore

      self.save_name   = self.net_name + '.save'
      self.ckpt_name   = self.net_name + '.ckpt'
      self.savestr     = FLAGS.run_dir + self.ckpt_name
      self.filestr     = FLAGS.run_dir + 'tensorlogs/' + timestr + '/' + self.net_name + '/'
      self.logstr      = self.filestr  + self.ckpt_name

      with tf.device('/gpu:3'):
        init_scope_vars()
        print("\rSETTING UP %s INPUTS"%self.net_name,end='')
        self.inputs()
        print("\rSETTING UP %s INFERENCE"%self.net_name,end='')
        self.inference()
        print("\rSETTING UP %s METRICS"%self.net_name,end='')
        self.metrics()

      print("\rINITIALIZING %s NETWORK"%self.net_name,end='')
      self.sess.run(tf.local_variables_initializer())
      self.sess.run(tf.global_variables_initializer())
      print("\rSETTING UP %s SAVER"%self.net_name,end='')
      self.saver()
      self.step        = tf.train.global_step(self.sess,self.global_step)
      util.get_params()
  # END __init__

  def inputs(self):
    # Placeholder for Images
    self.images = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.imgH,FLAGS.imgW,3])
    # Placeholder for Labels
    self.labels = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.imgH,FLAGS.imgW,1])
    # Generating our count
    self.count  = tf.reduce_sum(self.labels)
    tf.summary.image("Image",self.images)
    util.heat_map_log("Label",self.labels,FLAGS.imgH,FLAGS.imgW)
    tf.summary.scalar("Count",self.count)
  # END inputs

  def inference(self):
    with tf.variable_scope('Image_Ops') as scope:
      self.img_pat  = Image_To_Patch(self.images)
      self.lab_pat  = Image_To_Patch(self.labels)

    with tf.variable_scope("Inference") as scope:
      net = self.images / 255
      net = ops.conv2d(net, 16, kernel = 7)
      net = ops.batch_norm(net)
      net = ops.relu(net)
      net = ops.max_pool(net,4,4)

      net = ops.conv2d(net, 32, kernel = 6)
      net = ops.batch_norm(net)
      net = ops.relu(net)
      net = ops.max_pool(net,4,4)

      net = ops.conv2d(net, 48, kernel = 5)
      net = ops.batch_norm(net)
      net = ops.relu(net)
      net = ops.max_pool(net,4,4)

      net = ops.conv2d(net, 64, kernel = 4)
      net = ops.batch_norm(net)
      net = ops.relu(net)
      net = ops.max_pool(net,2,2)

      net = ops.conv2d(net, 2, kernel = 1)
      self.logits = net
  # END inference

  def metrics(self):
    with tf.variable_scope('Discriminatory_Metrics') as scope:
      disc_lab   = disc_label_gen(self.lab_pat)
      disc_log   = self.logits

      num_pat    = FLAGS.imgH // FLAGS.patch_size * FLAGS.imgW // FLAGS.patch_size

      # Do something with this disc_lab to make it 2d instead of 1d.

      patch_perc = tf.reduce_sum(disc_lab) / num_pat
      tf.summary.scalar("PPer",patch_perc)
      tf.summary.scalar("CPer",tf.reduce_mean(tf.cast(util.threshold(self.labels),tf.float32)))
      acc      = ops.accuracy(disc_lab,tf.argmax(self.logits,-1))

      # Generate our weights.
      weight   = tf.sqrt(1 / patch_perc)
      loss     = ops.weighted_xentropy_loss(disc_lab,disc_log,weight)

    with tf.variable_scope('Optimizer') as scope:
      # During training, optimize over the loss, oterwise just increment the
      #   step by one.
      self.train = self.optomize(loss) if self.training else tf.assign_add(self.global_step,1)

    with tf.variable_scope('Formatting') as scope:
      self.logs = tf.argmax(self.logits,-1)

      # Log our label and log matrices for visualization
      pat_shape = [1,FLAGS.imgH // FLAGS.patch_size,FLAGS.imgW // FLAGS.patch_size,1]
      tf.summary.image("Pat_Lab",tf.cast(tf.reshape(disc_lab ,pat_shape) ,tf.float32))
      tf.summary.image("Pat_Log",tf.cast(tf.reshape(self.logs,pat_shape) ,tf.float32))

      # Reshape our labels and logits to 1d arrays for ease of seeing. The
      #  patches are extracted into [PatchNum][PatchHeight][PatchWidth], and
      #  are accessed outside by referencing the patch number corresponding to
      #  positive labels.
      disc_log = tf.reshape(self.logs,(num_pat,1))
      disc_lab = tf.reshape(disc_lab ,(num_pat,1))

      # The next few lines of code generate the patch labels that will be used
      #  for the counting network. During training, it uses the ground truth
      #  labels, and during testing it uses the predicted patch labels
      lab_loc = tf.logical_and(    self.training ,tf.equal(disc_lab,1))
      log_loc = tf.logical_and(not self.training ,tf.equal(disc_log,1))
      self.logs = tf.logical_or(lab_loc,log_loc)
      self.logs = tf.cast(self.logs,tf.uint8)

    with tf.variable_scope("Losses") as scope:
      tf.summary.scalar("disc_Xent_loss",loss)

    self.metrics = [self.train,loss,acc]

  def optomize(self,loss,learning_rate = .001):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      with tf.variable_scope("Optimizer") as scope:
        if FLAGS.lr_decay:
          learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,100, 0.90, staircase=True)
          tf.summary.scalar("Learning_Rate",learning_rate)
        optomizer = tf.train.AdamOptimizer(learning_rate,epsilon = 1e-5)
        # optomizer = tf.train.RMSPropOptimizer(learning_rate,decay = 0.9, momentum = 0.3)
        train     = optomizer.minimize(loss,self.global_step)
        return train

  def saver(self):
    self.saver     = tf.train.Saver()
    self.summaries = tf.summary.merge_all()
    self.writer    = tf.summary.FileWriter(self.logstr,self.sess.graph)

    if self.restore or not self.training:
      print('\rRESTORING %s NETWORK'%self.net_name)
      self.saver.restore(self.sess,tf.train.latest_checkpoint(FLAGS.run_dir,latest_filename = self.save_name))
      if not self.training:
        self.sess.run(tf.assign(self.global_step,0))
    # END SAVER

  def save(self,step = None):
    if self.training:
      self.saver.save(self.sess,self.savestr,global_step = step,latest_filename = self.save_name)
    self.saver.save(self.sess,self.logstr ,global_step = step,latest_filename = self.save_name)
  # END SAVE

  def run(self):
    t_op = [self.images,self.labels,self.img_pat,self.lab_pat,self.logs,self.metrics,self.count]
    op   = [self.images,self.labels,self.img_pat,self.lab_pat,self.logs,self.metrics,self.summaries,self.count]
    self.step += 1

    imgs,labs,ids = self.generator.get_next_batch(1)
    labs = np.expand_dims(np.array(labs),-1)
    fd   = {self.images:imgs, self.labels:labs}

    images,_img_pat,_lab_pat,_logs = 0,0,1,2

    if self.step % 10 == 0 or not self.training:
      images,labels,_img_pat,_lab_pat,_logs,metrics,_summ_result,_count = self.sess.run(op,feed_dict = fd)
      if FLAGS.adv_logging:
        self.writer.add_run_metadata(run_metadata,'step%d'%self.step)
      self.writer.add_summary(_summ_result,self.step)
      if self.step % 100 == 0 and self.training:
        self.save(self.step)
    else:
      images,labels,ids[0],_img_pat,_lab_pat,_logs,metrics,_count = self.sess.run(t_op,feed_dict = fd)
    return images,labels,ids[0],_img_pat,_lab_pat,_logs,_count,metrics[1]
  # END RUN

  def close(self):
    if self.training:
      self.sess.run(tf.assign(self.global_step,self.step))
    self.save(self.step)
    self.sess.close()
  # END CLOSE
