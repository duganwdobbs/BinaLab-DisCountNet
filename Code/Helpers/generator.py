from PIL import Image
import numpy as np, tensorflow as tf
import os, random, pickle, json, cv2

# Basic model parameters as external flags.
imgW = int(224)
imgH = int(224)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('imgW'       , 2048                                ,'Image Width')
flags.DEFINE_integer('imgH'       , 1536                                ,'Image Height')
flags.DEFINE_integer('gauss_size' , 51                                 ,'Gaussian Kernel Size')
flags.DEFINE_integer('gauss_sigma' , 9                                 ,'Gaussian Kernel Sigma')
flags.DEFINE_integer('num_epochs' , 5                                  ,'Number of epochs to run trainer.')


class DataGenerator:
  def __init__(self,split,base_directory):
    self.base_directory = base_directory + '/'
    self.img_directory  = base_directory + '/Images/'
    self.json_directory = base_directory + '/metadata/'

    self.img_ext = '.JPG'

    self.split = split

    # If split lists don't exist as files, create them. Shuffle values, then
    # write to test,train,val.lst
    if (not os.path.isfile(base_directory + '/TEST.lst' ) or
        not os.path.isfile(base_directory + '/TRAIN.lst')   ):

      print("\rFILE SPLITS NOT FOUND... REBUILDING.")

      # Find all files
      file_list = [f.replace(self.img_ext,'') for f in os.listdir(self.img_directory ) if f.endswith(self.img_ext) and not f.startswith('.')]

      random.shuffle(file_list)
      # Train only on segmentation available data
      split_perc = len(file_list) * 7 // 10
      train_list = file_list[:split_perc]
      test_list  = file_list[split_perc:]
      # Test on everything else
      print("\r%d FILES FOUND                                "%(len(file_list)),end='')

      self.file_writer('TRAIN',train_list)
      self.file_writer('TEST' ,test_list )

    # Assign internal list
    self.internal_list = self.file_reader(base_directory, split)

    # Shuffle internal list order (Random Shuffle Batch)
    if self.split == 'TRAIN':
      random.shuffle(self.internal_list)
    else:
      self.internal_list.sort()

    # Necessary values: num_examples (len(list))
    self.num_examples = len(self.internal_list)
    #                   num_seen     (0)
    self.num_seen     = 0
    self.epoch        = 0

  def file_writer(self,split,list):
    with open(self.base_directory + split + '.lst','wb') as fp:
      pickle.dump(list,fp)

  def file_reader(self,directory,split):
    with open(self.base_directory + split + '.lst','rb') as fp:
      list = pickle.load(fp)
    return list

  def get_next_batch(self,batch_size):
    # If batch_zie + num_seen > num_examples, just return
    #   examples to end of list.
    if batch_size+self.num_seen > self.num_examples:
      self.epoch += 1
      self.num_seen = 0
      random.shuffle(self.internal_list)
      if self.epoch >= FLAGS.num_epochs:
        print("\rEpoch %d of %d done."%(self.epoch,FLAGS.num_epochs))
        raise IndexError
    batch_list = self.internal_list[self.num_seen:self.num_seen+batch_size]
    imgs,anns  = self.load_imgs_annotations(batch_list)
    self.num_seen += batch_size
    return imgs,anns,batch_list

  def load_imgs_annotations(self,img_ids):
    imgs = []
    anns = []
    seqs = []
    for img_id in img_ids:
      # Load Image, CV2 plays better with shapes,
      #   NOTE: loads in BGR, not RGB.
      #   NOTE: CV2 operates in [W][H][C], not [H][W][C]
      # print("Loading ",self.img_directory  + img_id + self.img_ext)
      img = cv2.imread(self.img_directory  + img_id + self.img_ext )
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      imgH,imgW,imgC = img.shape

      # Calculate the necessary dimension scalaing to obtain our target size
      #   NOTE: It will be necessary to use these to shift our GT pixel location
      fx,fy = FLAGS.imgW/ imgW, FLAGS.imgH / imgH

      # Resize Image
      img = cv2.resize(img,None,fx = fx,fy = fy)

      # The GT annotations are actually labeled in respect to the images resized
      #   to 4000x3000, so we need that fx, fy

      points = self.load_json_data(self.json_directory + img_id + '.json',fx,fy)

      # Single Point Annotatations
      ann = np.zeros((FLAGS.imgH,FLAGS.imgW,1))
      for point in points:
        py,px = point
        ann[int(py),int(px)] = 1

      # Counting GT annotations
      ann = cv2.GaussianBlur(ann,(FLAGS.gauss_size,FLAGS.gauss_size),FLAGS.gauss_sigma)

      imgs.append(img)
      anns.append(ann)

    return imgs,anns

  def load_json_data(self,json_file, fx,fy):
    with open(json_file) as f:
      meta = json.load(f)
      points = meta['points']
      for n in range(len(points)):
        x,y = points[n]
        x = min(max(fx * x,0),FLAGS.imgW-1)
        y = min(max(fy * y,0),FLAGS.imgH-1)
        points[n] = (y,x)
      return points

  def test(self):
    import matplotlib.pyplot as plt
    while True:
      imgs,anns,batch_list = self.get_next_batch(1)
      # print("Annotations: ",anns)
      # print("IDs: ",batch_list)
      for x in range(len(imgs)):
        img = imgs[x]
        ann = anns[x]
        print(img.shape)
        print(ann.shape)
        # plt.subplot(121),plt.imshow(img),plt.title("Image")
        # plt.subplot(122),plt.imshow(ann),plt.title("Annot")
        # plt.show()

def calc_max_gauss_val(gauss_size,gauss_sigma):
  mat = np.zeros((gauss_size*3,gauss_size*3))
  mat[gauss_size*3//2,gauss_size*3//2] = 2
  mat = cv2.GaussianBlur(mat,(gauss_size,gauss_size),gauss_sigma)
  return np.amax(mat)

def perturb(imgs_labs_losses):
  for x in range(len(imgs_labs_losses)):
    img,lab,loss = imgs_labs_losses[x]

    flipud = random.randint(0,1)
    fliplr = random.randint(0,1)
    rot  = random.randint(0,3)

    if flipud == 1:
      img = np.flipud(img)
      lab = np.flipud(lab)
    if fliplr == 1:
      img = np.fliplr(img)
      lab = np.fliplr(lab)
    while rot > 0:
      rot -= 1
      img = np.rot90(img)
      lab = np.rot90(lab)

    imgs_labs_losses[x] = (img,lab,loss)
  return imgs_labs_losses


if __name__ == '__main__':
  print("STARTING GENERATOR TEST")
  generator = DataGenerator('TRAIN','D:/Binalab-Animal-detection-and-counting/Data')
  generator.test()
  generator.test()
