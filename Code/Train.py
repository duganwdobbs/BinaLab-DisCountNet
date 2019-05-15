# Purpose of this file:
# Setup Environment
# Define networks
# Run training by telling networks to train until stopping point.
# Manage any multiprocessing environment

import timeit

import os
import time
import h5py
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
  flags.DEFINE_string ('base_dir'  ,'/data1/ddobbs/Binalab-Animal-detection-and-counting/Data','Base os specific DIR')

# These values affect training, and can be set accordingly.
flags.DEFINE_boolean('l2_loss'    , False                              ,'If we use l2 regularization')
flags.DEFINE_boolean('lr_decay'   , False                              ,'If we use Learning Rate Decay')
flags.DEFINE_integer('batch_size' , 1                                  ,'Batch size for training.')
flags.DEFINE_boolean('adv_logging', False                              ,'If we log metadata and histograms')

# If you have different sized images or more than two classes, these can be
#   changed here.
flags.DEFINE_integer('patch_size' , 128                                ,'Patch Size')
flags.DEFINE_integer('num_classes', 2                                  ,'Number of classes.')

# A pre-calculated thresholded value, depricated in final implementation, but
#   left in the code to maintain consistency
maxval = generator.calc_max_gauss_val(FLAGS.gauss_size,FLAGS.gauss_sigma)
flags.DEFINE_float(  'gt_max'     ,maxval                              ,'Value to threshold the labels')
flags.DEFINE_float(  'thresh_val' , .0002                              ,'Value to threshold the labels')

# Some basic directories.
flags.DEFINE_string ('run_dir'    , FLAGS.base_dir  + '/network_log/'  ,'Location to store the Tensorboard Output')
flags.DEFINE_string ('train_dir'  , FLAGS.base_dir  + '/'              ,'Location of the tfrecord files.')
flags.DEFINE_string ('net_name'   , 'DisCountNet'                           ,'Location of the tfrecord files.')
flags.DEFINE_string ('ckpt_name'  , 'DisCountNet' + '.ckt'            ,'Checkpoint name')
flags.DEFINE_string ('ckpt_i_name', 'DisCountNet' + '-interrupt.ckpt','Interrupt Checkpoint name')

# These options evolved into the discriminator. The standard method is when both
#   of these options are true.
flags.DEFINE_bool("cell_selection"   ,True,"If we use cell selection optimization")
flags.DEFINE_bool("positive_training",True ,"If we only train on positive patches.")

# This functions receives patches, and translates them back into a full image on
#   the CPU. When timed, this operation took approximetly the same time on CPU
#   as GPU.
def patches_to_image(patch):
  imgH,imgW = FLAGS.imgH,FLAGS.imgW
  patH,patW = FLAGS.patch_size,FLAGS.patch_size
  imgSize = [imgH,imgW]
  img_re = np.reshape  (patch, [int(imgH / patH), int(imgW / patW), patH, patW])
  img_tr = np.transpose(img_re,[0,2,1,3])
  image  = np.reshape  (img_tr,imgSize)
  return image

def test(train_run = (False,False), restore = True, epoch = 0):
  return train(train_run,restore,epoch)

def train(train_run = (True,True), restore = False, epoch = 0):
  disc_train_run,segm_train_run = train_run
  train_run = disc_train_run or segm_train_run

  if not train_run:
    FLAGS.batch_size = 1
    FLAGS.num_epochs = 1

  timestr        = time.strftime("TRAIN/%d_%b_%Y_%H_%M",time.localtime()) if train_run else time.strftime("TEST/%d_%b_%Y_%H_%M",time.localtime())
  timestr = timestr + "_EPOCH_%d"%epoch
  # Location to log to.
  split = 'TRAIN' if train_run else 'TEST'

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
  sum_cmat = np.zeros((FLAGS.num_classes,FLAGS.num_classes))

  game_1 = []
  game_2 = []
  game_3 = []
  game_4 = []
  ssim   = []
  psnr   = []
  rers   = []
  maes   = []
  mses   = []

  try:
    while not coord.should_stop():
      # Run the network and write summaries
      start = timeit.default_timer()
      try:
        orig_img,orig_lab,ids,img_pat,lab_pat,disc_log,count,disc_loss = discriminator.run()
        disc_losses.append(disc_loss)
      except IndexError:
        break
        # Epoch done.
      if FLAGS.cell_selection or (FLAGS.positive_training and train_run):
        img_pat  = np.reshape(img_pat,(-1,FLAGS.patch_size,FLAGS.patch_size,3))
        lab_pat  = np.reshape(lab_pat,(-1,FLAGS.patch_size,FLAGS.patch_size,1))
        disc_log = np.squeeze(disc_log)
      else:
        img_pat  = np.reshape(orig_img,(-1,FLAGS.imgH,FLAGS.imgW,3))
        lab_pat  = np.reshape(orig_lab,(-1,FLAGS.imgH,FLAGS.imgW,1))
        disc_log = np.ones((FLAGS.batch_size))
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
        seg_log,seg_loss,per_pat_loss,cmat = segmenter.run(new_imgs,new_labs,count)
        segm_losses.append(seg_loss)
        sum_cmat += cmat

        # Train on the worst 1/2 images twice.
        if train_run:
          im_loss = zip(new_imgs,new_labs,per_pat_loss)
          [imgs_labs_losses.append(im_lab_loss) for im_lab_loss in im_loss]


        # Do some more processing, weave the resultant patches into an array
        # of the resultant logit map
        y = 0
      stop = timeit.default_timer()
      print('\rTime: ', stop - start,end='')

      if not train_run:
        full_pats = np.zeros(shape = lab_pat.shape,dtype = np.float32)
        for x in range(len(disc_log)):
          if(disc_log[x] == 1):
            full_pats[x] = seg_log[y]
            y+=1

        orig_img = np.squeeze(orig_img).astype('uint8')
        orig_lab = np.squeeze(orig_lab)
        result = np.squeeze(patches_to_image(full_pats))

        with h5py.File(filestr + '%s_log.png'%ids, 'w') as hf:
            hf['density'] = np.squeeze(orig_lab)
        with h5py.File(filestr + '%s_ann.png'%ids, 'w') as hf:
            hf['density'] = np.squeeze(result)

        # Before we colorize the result, we want to run the GAME metrics
        game_1.append(util.cpu_GAME(orig_lab,result,1))
        game_2.append(util.cpu_GAME(orig_lab,result,2))
        game_3.append(util.cpu_GAME(orig_lab,result,3))
        game_4.append(util.cpu_GAME(orig_lab,result,4))
        ssim.append(util.cpu_psnr(orig_lab,result))
        psnr.append(util.cpu_ssim(orig_lab,result))
        rers.append(np.abs(np.sum(orig_lab)-np.sum(result))/np.sum(orig_lab))
        maes.append(np.abs(np.sum(orig_lab)-np.sum(result)))
        mses.append((np.sum(orig_lab)-np.sum(result))**2)

        result = util.cpu_colorize( result )
        orig_lab = util.cpu_colorize(np.squeeze(orig_lab))

        img = Image.fromarray(result)
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
          img = Image.fromarray(np.squeeze(util.cpu_colorize( seg_log[x])))
          img.save(filestr + 'patches/' + '%d_%d_log_pat.png'%(step,x))
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
    else:
      print("GAME 1,2,3,4")
      print("Game 1, ",np.mean(game_1))
      print("Game 2, ",np.mean(game_2))
      print("Game 3, ",np.mean(game_3))
      print("Game 4, ",np.mean(game_4))
      print("SSIM, ",np.mean(ssim))
      print("PSNR, ",np.mean(psnr))
      print("Rel_Err, ",np.mean(rers))
      print("MAE, ",np.mean(maes))
      print("MSE, ",np.mean(mses))
    coord.request_stop()
  coord.join(threads)

  np.save(filestr + "data.dat",sum_cmat)

  discriminator.close()
  segmenter.close()
  return np.mean(disc_losses),np.mean(segm_losses)

def main(_):
  disc_training = True
  segm_training = True

  # Number of epochs to overlap before stopping training if loss does not
  #   improve
  max_overlap = 5

  # Number of current overlaps
  disc_cur_lap = 0
  segm_cur_lap = 0

  # A variable to hold the best loss value, init at a large value
  disc_best_loss = 1e6
  segm_best_loss = 1e6

  # A variable to hold this loss
  disc_this_loss = 1e5
  segm_this_loss = 1e5

  # A variable to hold the current epoch
  cur_epoch = 0

  # A variable to hold the best epoch
  disc_best_epoch = 0
  segm_best_epoch = 0

  while ((disc_this_loss < disc_best_loss or disc_cur_lap < max_overlap) or
         (segm_this_loss < segm_best_loss or segm_cur_lap < max_overlap)):

    if disc_cur_lap >= max_overlap:
      disc_training = False

    if segm_cur_lap >= max_overlap:
      segm_training = False

    # If this loss is better than the best loss, do some stuff.
    if disc_this_loss <= disc_best_loss and disc_training:
      disc_best_loss = disc_this_loss
      disc_cur_lap   = 0
      disc_best_epoch= cur_epoch
    else:
      disc_cur_lap += 1

    if segm_this_loss <= segm_best_loss:
      segm_best_loss = segm_this_loss
      segm_cur_lap   = 0
      segm_best_epoch= cur_epoch
    else:
      segm_cur_lap += 1
    # Otherwise, the overlap increases
    # Increment the epoch, so this will start us at 1.
    cur_epoch += 1

    if cur_epoch != 1:
      print("Disc This Epoch: %d, Best Epoch: %d, This Loss: %.4e, Best Loss %.4e, Overlap: %d/%d"%(cur_epoch-1,disc_best_epoch,disc_this_loss,disc_best_loss,disc_cur_lap,max_overlap))
      print("Segm This Epoch: %d, Best Epoch: %d, This Loss: %.4e, Best Loss %.4e, Overlap: %d/%d"%(cur_epoch-1,segm_best_epoch,segm_this_loss,segm_best_loss,segm_cur_lap,max_overlap))

    # We wish to restore if we are past the first epoch.
    restore = (cur_epoch > 1)

    train(train_run = (disc_training,segm_training) , restore = restore, epoch = cur_epoch)
    disc_this_loss,segm_this_loss = test(train_run = (False,False), restore = True, epoch = cur_epoch)

    if disc_cur_lap >= max_overlap and segm_cur_lap > max_overlap:
      break


if __name__ == '__main__':

  print(tf.app.flags.FLAGS)
  input("Press ENTER to continue...")

  tf.app.run()
