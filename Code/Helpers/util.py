# Data Testing, makes image label pairs in reasonable resolutions.
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.cm as cm
import math
from skimage.measure import compare_ssim

flags = tf.app.flags
FLAGS = flags.FLAGS

def get_params():
  total_parameters = 0
  for variable in tf.trainable_variables():
    local_parameters=1
    shape = variable.get_shape()  #getting shape of a variable
    for i in shape:
        local_parameters*=i.value  #mutiplying dimension values
    total_parameters+=local_parameters
  print('\nParameters: %d'%total_parameters)

def get_Im_Specs():
  imgW    = FLAGS.imgW
  imgH    = FLAGS.imgH
  patW    = FLAGS.patch_size
  patH    = FLAGS.patch_size
  num_pat = (imgW * imgH) / (patW * patH)
  return int(imgW),int(imgH),int(patW),int(patH),int(num_pat)

def Image_To_Patch(image):
  with tf.variable_scope("Image_To_Patch") as scope:
    imgW,imgH,patW,patH,numP = get_Im_Specs()
    chan = image.shape[-1].value
    image = tf.squeeze(image)
    image = tf.reshape(image,[FLAGS.batch_size,imgH,imgW,chan])
    patSize = [1,patH,patW,1]

    patches      = tf.extract_image_patches(image,patSize,patSize,[1,1,1,1],'VALID')
    # reshape the patches to the correct dimensions.
    # TODO: PROBLEM HERE
    patches      = tf.reshape(patches,[FLAGS.batch_size,imgH//patH,imgW//patW,patH,patW,chan])
    # tf.summary.image('Patches',patches)
    return patches

def ImSizeToPatSize(image):
  blkH = FLAGS.patch_size
  blkW = FLAGS.patch_size
  block_shape = (blkH,blkW)
  return block_shape

def Patch_To_Image(patch):
  with tf.variable_scope("Patch_To_Image") as scope:
    imgW,imgH,patW,patH,numP = get_Im_Specs()
    patch = tf.squeeze(patch)
    chan = tf.minimum(1,patch.shape[-1].value)
    imgSize = [1,imgH,imgW,chan]
    patSize = [-1,patH,patW,chan]
    patch = tf.reshape(patch,patSize)

    img_re = tf.reshape  (patch, [int(imgH / patH), int(imgW / patW), patH, patW])
    img_tr = tf.transpose(img_re,[0,2,1,3])
    image  = tf.reshape  (img_tr,imgSize)

    # tf.summary.image('P->I',image)
    return image

def threshold(net,thresh_val = None,min_val = 0,max_val = 1):
  if thresh_val is None:
    try:
      thresh_val = FLAGS.thresh_val
    except:
      assert("Threshold value not given.")

  with tf.variable_scope("Threshold") as scope:
    mins = tf.ones_like(net,tf.float32) * min_val
    maxs = tf.ones_like(net,tf.float32) * max_val
    # Shift our threshold values to 0 and 1
    net = tf.cast(net,tf.float32)
    pos = tf.greater(net,thresh_val)
    net = tf.where(pos,maxs,mins)
  return net

# Generates True / False labels in the shape of [#Batch][2], where 0 is false,
#   and 1 is true. NOTE: Only works in binary classificaitons
def disc_label_gen(label_patches):
  with tf.variable_scope('Discriminator_Label_Gen') as scope:
    label_patches = threshold(label_patches)
    # Incoming Shape: [B,H/p,W/p,p,p,1]
    patch_labels  = tf.reduce_sum(label_patches,(3,4))
    # This represents the number of important pixels per patchs
    # Outgoing Shape: [B,H/p,W/p,1]
    patch_labels = threshold(patch_labels)
    # This thresholded value represents the label patch, 1 or 0.
    return patch_labels

# Generates True / False labels in the shape of [#Batch][2], where 0 is false,
#   and 1 is true. NOTE: Only works in binary classificaitons
def count_label_gen(label_patches):
  with tf.variable_scope('Discriminator_Label_Gen') as scope:
    # Incoming Shape: [B,p,p,1]
    patch_sums   = tf.reduce_sum(label_patches,(1,2))
    # Outgoing Shape: [B,H/p,W/p,1]
    patch_labels = threshold(patch_sums)
    return patch_labels

# USING FROM https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b
#   CREDIT TO jimfleming, https://github.com/jimfleming
def colorize(value, vmin=0, vmax=.004, cmap='bwr'):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'bwr')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```

    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    map = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = map(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value

def cpu_colorize(value,vmin=0,vmax=None,cmap='bwr'):
  vmin = np.amin(value) if vmin is None else vmin
  vmax = FLAGS.gt_max if vmax is None else vmax
  value = (value - vmin) / (vmax - vmin) # vmin..vmax

  # squeeze last dim if it exists
  value = np.squeeze(value)

  # quantize
  value = (value * 255).astype(np.uint8)

  # gather
  map = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
  value = (map(value) * 255).astype(np.uint8)

  return value

def heat_map_log(name,img,imgH,imgW):
  img = colorize(img)
  img = tf.squeeze(img)
  img = tf.reshape(img,[-1,imgH,imgW,3])
  tf.summary.image(name,img)

def cpu_psnr(img1, img2):
  mse = np.mean( (img1 - img2) ** 2 )
  if mse == 0:
    return 100
  PIXEL_MAX = max(np.amax(img2),np.amax(img1))
  return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def cpu_ssim(img1,img2):
  return compare_ssim(img1,img2)

def cpu_GAME(labels,logits,n):
  abs_err = np.abs(labels-logits)

  h,w = labels.shape
  num_gd  = 2**n
  region_h = h / num_gd
  region_w = w / num_gd
  regions = []
  for y in range(num_gd):
    for x in range(num_gd):
      # [[B,H,W,C]]
      regions.append(abs_err[
                  int(region_h*y):int(region_h*(y+1)),
                  int(region_w*x):int(region_w*(x+1))  ])
  # [4**n,B,H,W,C,]
  regions = np.array(regions)
  # -> [B,4**n]
  # input(regions.shape)
  regions = np.sum(regions,(1,2))
  game_n  = np.mean(regions)
  return game_n
