# Purpose of this file:
# Setup Environment
# Define networks
# Run training by telling networks to train until stopping point.
# Manage any multiprocessing environment

import os
import time
import platform

#Custom Imports=
import Helpers.filetools as ft
import Helpers.util as util

import Helpers.generator as generator


from discnet  import DiscNet
from segnet   import SegNet

# Aliased Imports
import numpy         as     np
import tensorflow    as     tf
from PIL             import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS

if   platform.system() == 'Windows':
  flags.DEFINE_string ('base_dir'  ,'D:/Binalab-Animal-detection-and-counting/Data','Base os specific DIR')
elif platform.system() == 'Linux':
  flags.DEFINE_string ('base_dir'  ,'/data0/ddobbs/Binalab-Animal-detection-and-counting/','Base os specific DIR')

flags.DEFINE_boolean('l2_loss'    , False                              ,'If we use l2 regularization')
flags.DEFINE_boolean('lr_decay'   , False                              ,'If we use Learning Rate Decay')
flags.DEFINE_boolean('adv_logging', False                              ,'If we log metadata and histograms')

flags.DEFINE_integer('batch_size' , 1                                  ,'Batch size for training.')

flags.DEFINE_integer('patch_size' , 128                                ,'Patch Size')
flags.DEFINE_float(  'thresh_val' , .0002                              ,'Value to threshold the labels')

maxval = generator.calc_max_gauss_val(FLAGS.gauss_size,FLAGS.gauss_sigma)
flags.DEFINE_float(  'gt_max'     ,maxval                              ,'Value to threshold the labels')

flags.DEFINE_string ('run_dir'    , FLAGS.base_dir  + '/network_log/'  ,'Location to store the Tensorboard Output')
flags.DEFINE_string ('mod_dir'   , FLAGS.base_dir  + '/saved_model/'   ,'Location to store the best Saved Model')
flags.DEFINE_string ('train_dir'  , FLAGS.base_dir  + '/'              ,'Location of the tfrecord files.')
flags.DEFINE_string ('net_name'   , 'CowNet'                           ,'Location of the tfrecord files.')
flags.DEFINE_string ('ckpt_name'  , 'CowNet' + '.ckt'            ,'Checkpoint name')
flags.DEFINE_string ('ckpt_i_name', 'CowNet' + '-interrupt.ckpt','Interrupt Checkpoint name')

def patches_to_image(patch):
  imgH,imgW = FLAGS.imgH,FLAGS.imgW
  patH,patW = FLAGS.patch_size,FLAGS.patch_size
  imgSize = [imgH,imgW]
  img_re = np.reshape  (patch, [int(imgH / patH), int(imgW / patW), patH, patW])
  img_tr = np.transpose(img_re,[0,2,1,3])
  image  = np.reshape  (img_tr,imgSize)
  image  = image * 255
  image  = image.astype('uint8')
  return image

def test(train_run = (False,False), restore = True, epoch = 0):
  return train(train_run,restore,epoch)

def train(train_run = (True,True), restore = False, epoch = 0):
  disc_train_run,segm_train_run = train_run
  train_run = disc_train_run or segm_train_run

  if not train_run:
    FLAGS.batch_size = 1
    FLAGS.num_epochs = 1

  train_run = disc_train_run or segm_train_run

  timestr        = time.strftime("TRAIN/%d_%b_%Y_%H_%M",time.localtime()) if train_run else time.strftime("TEST/%d_%b_%Y_%H_%M",time.localtime())
  timestr = timestr + "_EPOCH_%d"%epoch
  # Location to log to.
  split = 'TRAIN' if train_run else 'TEST'

  # We just run from the saved model directory for demoing.
  FLAGS.run_dir = FLAGS.mod_dir
  filestr        = FLAGS.run_dir + "tensorlogs/" + timestr + '/'
  ft.directoryFixer(filestr + 'patches/')

  print("Running from: " + filestr)

  tf.reset_default_graph()
  discriminator = DiscNet(disc_train_run,restore,timestr,split)

  tf.reset_default_graph()
  segmenter     = SegNet(segm_train_run,restore,timestr)

  # Starts the input generator
  print("\rSTARTING INPUT GENERATION THREADS...")
  coord          = tf.train.Coordinator()
  threads        = tf.train.start_queue_runners(sess = discriminator.sess, coord = coord)
  print("STARTING TRAINING...")

  step = 0
  full_pats = 0
  epoch = 0

  disc_losses = []
  segm_losses = []

  try:
    while not coord.should_stop():
      # Run the network and write summaries
      try:
        orig_img,orig_lab,img_pat,lab_pat,disc_log,count,disc_loss = discriminator.run()
        disc_losses.append(disc_loss)
      except IndexError:
        break
        # Epoch done.
      img_pat  = np.reshape(img_pat,(-1,FLAGS.patch_size,FLAGS.patch_size,3))
      lab_pat  = np.reshape(lab_pat,(-1,FLAGS.patch_size,FLAGS.patch_size,1))
      disc_log = np.squeeze(disc_log)


      # Do some processing, create new array with only patches we need.
      new_imgs = []
      new_labs = []

      for x in range(len(disc_log)):
        if(disc_log[x] == 1):
          new_imgs.append(img_pat[x])
          new_labs.append(lab_pat[x])

      new_imgs = np.array(new_imgs)
      new_labs = np.array(new_labs)

      imgs_labs_losses   = []

      if(np.sum(disc_log) > 0):
        seg_log,seg_loss,per_pat_loss = segmenter.run(new_imgs,new_labs,count)
        segm_losses.append(seg_loss)

        # Train on the worst 1/2 images twice.
        if train_run:
          im_loss = zip(new_imgs,new_labs,per_pat_loss)
          [imgs_labs_losses.append(im_lab_loss) for im_lab_loss in im_loss]


        # Do some more processing, weave the resultant patches into an array
        # of the resultant logit map
        y = 0



      if not train_run:
        full_pats = np.zeros(shape = lab_pat.shape)
        for x in range(len(disc_log)):
          if(disc_log[x] == 1):
            full_pats[x] = seg_log[y]
            y+=1

        orig_img = np.squeeze(orig_img).astype('uint8')
        orig_lab = util.cpu_colorize(np.squeeze(orig_lab))
        # Go from patches to full image logit map.
        result = patches_to_image(full_pats)
        img = Image.fromarray(util.cpu_colorize(result))
        img.save(filestr + '%d_log.png'%step)
        img = Image.fromarray(orig_img)
        img.save(filestr + '%d_img.png'%step)
        img = Image.fromarray(orig_lab)
        img.save(filestr + '%d_lab.png'%step)
        for x in range(new_imgs.shape[0]):
          img = Image.fromarray(np.squeeze(new_imgs[x]).astype(np.uint8))
          img.save(filestr + 'patches/' + '%d_%d_img_pat.png'%(step,x))
          img = Image.fromarray(np.squeeze(util.cpu_colorize(new_labs[x])))
          img.save(filestr + 'patches/' + '%d_%d_lab_pat.png'%(step,x))
          img = Image.fromarray(np.squeeze(util.cpu_colorize(seg_log[x])))
          img.save(filestr + 'patches/' + '%d_%d_lab_pat.png'%(step,x))
      step +=1

    # Train on bad patches over again.
    if segm_train_run:
      imgs_labs_losses.sort(key = lambda tup: tup[2])
      iterval = 0
      while len(imgs_labs_losses) > 1:
        iterval += 1
        imgs_labs_losses = imgs_labs_losses[len(imgs_labs_losses)//2:]
        generator.perturb(imgs_labs_losses)
        new_imgs = []
        new_labs = []
        [(new_imgs.append(new_img),new_labs.append(new_lab)) for new_img,new_lab,_ in imgs_labs_losses]
        # Run through 10 patches at a time as a batch.
        for x in range(len(new_imgs)//10 + 1):
          print('\rTRAINING ON HARD EXAMPLES %d/%d ITER %d'%(x,len(new_imgs)//10+1,iterval),end='')
          _ = segmenter.run(new_imgs[x*10:(x+1)*10],new_labs[x*10:(x+1)*10],0,False)
      print('\rDONE TRAINING HARD EXAMPLES')

  except KeyboardInterrupt:
    pass
  finally:
    if train_run:

      discriminator.save()
      segmenter.save()
    coord.request_stop()
  coord.join(threads)

  discriminator.close()
  segmenter.close()
  return np.mean(disc_losses),np.mean(segm_losses)

def main(_):

  disc_training = False
  segm_training = False

  test()


if __name__ == '__main__':
  tf.app.run()
