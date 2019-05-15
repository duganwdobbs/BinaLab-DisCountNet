import tensorflow as tf

from Helpers import ops,util

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
# DIRECTORIES
# TODO: SAVE IMAGES

class SegNet:
  def __init__(self,training,restore,timestr):
    with tf.Graph().as_default():
      self.net_name    = "SegNet"
      print("\rSETTING UP %s NETWORK"%self.net_name,end='')
      config = tf.ConfigProto(allow_soft_placement = True)
      config.gpu_options.allow_growth = True
      self.sess        = tf.Session(config = config)

      self.global_step = tf.Variable(0,name='global_step',trainable = False)
      self.training    = training
      self.restore     = restore

      self.levels = []

      self.ckpt_name   = self.net_name + '.ckpt'
      self.save_name   = self.net_name + '.save'
      self.savestr    = FLAGS.run_dir + self.ckpt_name
      self.filestr     = FLAGS.run_dir + 'tensorlogs/' + timestr + '/' + self.net_name + '/'
      self.logstr     = self.filestr  + self.ckpt_name

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
      self.step = tf.train.global_step(self.sess,self.global_step)
      util.get_params()

  # END __init__

  def inputs(self):
    self.imgs = tf.placeholder(tf.float32, shape = [None,FLAGS.patch_size,FLAGS.patch_size,3],name = 'Image_Patch')
    tf.summary.image("Patches",self.imgs)
    self.labs = tf.placeholder(tf.float32, shape = [None,FLAGS.patch_size,FLAGS.patch_size,1]  ,name = 'Label_Patch')
    self.lab_count = tf.placeholder(tf.float32, shape = ())

  # END INPUTS

  def inference(self):
    scales   = []
    strides  = [2,2,2,2]
    channels = 16
    with tf.variable_scope("Inference") as scope:
      net = self.imgs/ 255
      for stride in strides:
        net = ops.conv2d(net, channels, kernel = 3)
        net = ops.batch_norm(net)
        net = ops.relu(net)
        scales.append(net)
        net = ops.max_pool(net,2,2)
        channels += 16

      for scale in scales[::-1]:
        channels = channels // 2
        net = ops.deconv(net,channels,3,2)
        net = ops.delist([scale,net])

      net = ops.conv2d(net, 1, kernel = 1)
      # Sigmoid, and shift values to 0-1
      self.logs = tf.nn.sigmoid(net) * FLAGS.gt_max * 2

  # END INFERENCE

  def metrics(self):
    with tf.variable_scope('Segmentation_Metrics') as scope:
      count_seg_lab    = util.threshold(self.labs)
      tf.summary.image("SegLab",count_seg_lab * 255)
      count_seg_log    = util.threshold(self.logs)
      tf.summary.image("SegLog",count_seg_log * 255)
      patch_class_perc = tf.reduce_mean(tf.cast(count_seg_lab,tf.float32))
      tf.summary.scalar("PatchClassPercentage",patch_class_perc)

      util.heat_map_log("Lab_Pat",self.labs,FLAGS.patch_size,FLAGS.patch_size)
      util.heat_map_log("Log_Pat",self.logs,FLAGS.patch_size,FLAGS.patch_size)

      lab_count = tf.reduce_sum(self.labs)
      log_count = tf.reduce_sum(self.logs)

      tf.summary.scalar("Lab_Count",lab_count)
      tf.summary.scalar("Log_Count",log_count)

      abs_err = lab_count - log_count


      huber  = tf.losses.huber_loss(lab_count,log_count)
      mse_loss = ops.mse_loss(lab_count,log_count)

      weight = tf.sqrt(1/patch_class_perc)
      loss,l2= ops.l2loss(self.labs,self.logs,weight) #+ huber

      self.cmat = tf.confusion_matrix(tf.reshape(count_seg_lab,[-1]),tf.reshape(count_seg_log,[-1]),FLAGS.num_classes)

      acc  = ops.accuracy(count_seg_lab,count_seg_log)

    with tf.variable_scope("Losses") as scope:
      tf.summary.scalar("Segm_L2_Loss",loss)
      tf.summary.scalar("Abs_Err",abs_err)
      tf.summary.scalar("Rel_Err",tf.abs(abs_err)/lab_count)
      tf.summary.scalar("Huber_Err",huber)
      tf.summary.scalar("MSE_Err",mse_loss)

    with tf.variable_scope('Optimizer') as scope:
      self.train   = self.optomize(loss) if self.training else tf.assign_add(self.global_step,1)
    self.metrics = [self.train,loss,acc,l2,tf.abs(abs_err)]
  # END METRICS

  def optomize(self,loss,learning_rate = .001):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      with tf.variable_scope("Optimizer") as scope:
        learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,200, 0.90, staircase=True)
        # optomizer = tf.train.RMSPropOptimizer(learning_rate,decay = 0.9, momentum = 0.3)
        optomizer = tf.train.AdamOptimizer(learning_rate,epsilon = 1e-5)
        train     = optomizer.minimize(loss,self.global_step)
        return train
  # END OPTOMIZE

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

  def save(self,step=None):
    if self.training:
      self.saver.save(self.sess,self.savestr,global_step = step,latest_filename = self.save_name)
    self.saver.save(self.sess,self.logstr ,global_step = step,latest_filename = self.save_name)
  # END SAVE

  def run(self,images,labels,count,logging = True):
    t_op = [self.cmat,self.logs,self.metrics]
    op   = [self.logs,self.metrics,self.summaries]
    if logging:
      self.step += 1
    fd   = {self.imgs: images,self.labs: labels, self.lab_count:count}

    _logs = 0
    _loss = 0
    _cmat = 0
    _per_pat_loss = 0

    if self.step % 10 == 0 or not self.training:
      _logs,metrics,summaries = self.sess.run(op,feed_dict = fd)
      if logging:
        if FLAGS.adv_logging:
          self.writer.add_run_metadata(run_metadata,'step%d'%self.step)
        self.writer.add_summary(summaries,self.step)
      if self.step % 100 == 0 and self.training:
        self.save(self.step)
    else:
      _cmat,_logs,metrics = self.sess.run(t_op,feed_dict = fd)

    _loss = metrics[1]
    _per_pat_loss = metrics[3]
    return _logs,_loss, _per_pat_loss, _cmat
  # END RUN

  def close(self):
    if self.training:
      self.sess.run(tf.assign(self.global_step,self.step))
    self.save(self.step)
    self.sess.close()
  # END CLOSE
