from Helpers.generator import DataGenerator
from Helpers.util import cpu_colorize
from Helpers import filetools as ft

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import cv2

patW = 128
patH = 128

def Image_To_Patch(image,patH,patW):
  try:
    imgH,imgW,chan = image.shape
  except:
    imgH,imgW      = image.shape
  numH = imgH // patH
  numW = imgW // patW
  patches = []
  for y in range(numH):
    for x in range(numW):
      y_min = patH * y
      y_max = patH * (y+1)
      x_min = patW * x
      x_max = patW * (x+1)
      patches.append(image[y_min:y_max,x_min:x_max])
  return patches

def save_pats(img_arr,name,post,base_dir):
  for x in range(len(img_arr)):
    file_loc = base_dir+"/Visualize/%s/%d.png"%(name+post,x)
    ft.directoryFixer(file_loc.split('.')[0])
    img = Image.fromarray(img_arr[x])
    img.save(file_loc)

# Given image name, load GT
def get_patches(generator,image_name,base_dir):
  img,ann = generator.load_imgs_annotations([image_name])
  img = np.squeeze(np.array(img))
  ann = np.squeeze(np.array(ann))

  sav_img = Image.fromarray(img)
  sav_img.save(base_dir+'/Visualize/%s_%s.png'%(image_name,'IMG'))

  draw_sparse(base_dir,image_name,img,ann)
  gen_specs(ann,Image_To_Patch(ann,patW,patH))
  ann = cpu_colorize(ann,vmax = .005)

  img_pat = Image_To_Patch(img,patW,patH)
  save_pats(img_pat,image_name,'_IMG',base_dir)

  ann_pat = Image_To_Patch(ann,patW,patH)
  sav_img = Image.fromarray(ann)
  sav_img.save(base_dir+'/Visualize/%s_%s.png'%(image_name,'ANN'))
  save_pats(ann_pat,image_name,'_GT',base_dir)

def draw_sparse(base_dir,name,img,ann,post="_SPARSE"):
  imgs = Image_To_Patch(img,patW,patH)
  anns = Image_To_Patch(ann,patW,patH)
  try:
    imgH,imgW,chan = img.shape
  except:
    imgH,imgW      = img.shape
  numH = imgH // patH
  numW = imgW // patW


  print(img.shape[0:2])

  transparent = np.expand_dims(np.full(img.shape[0:2],.5),axis=2)
  transparent = np.concatenate([img,transparent],axis=2)
  opaque = np.expand_dims(np.full(img.shape[0:2],1),axis=2)
  opaque = np.concatenate([img,opaque],axis=2)
  # plt.imshow(img)
  # plt.show()
  # plt.imshow(ann)
  # plt.show()

  for y in range(numH):
    for x in range(numW):
      print(y,x,np.sum(anns[y*numW+x]))
      if np.sum(anns[y*numW+x]) < .1:
        y_min = patH * y
        y_max = patH * (y+1)
        x_min = patW * x
        x_max = patW * (x+1)
        img[y_min:y_max,x_min:x_max] = img[y_min:y_max,x_min:x_max] / 2 + 127
        # plt.imshow(img[y_min:y_max,x_min:x_max])
        # plt.show()

  print(opaque.shape)
  # img = Image.fromarray(opaque)
  # img.save(base_dir+"/Visualize/%s.png"%(name+post))
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
  cv2.imwrite(base_dir+"/Visualize/%s.png"%(name+post),img)

def gen_specs(ann,ann_pat):
  count = np.sum(ann)
  max   = np.amax(ann)
  pixs  = np.count_nonzero(ann)
  imgH,imgW = ann.shape
  numPi = imgH * imgW

  numPa = np.count_nonzero(np.sum(ann_pat,axis=(1,2)))
  PatPi = patW * patH * numPa


  print("Number of Pixels: %d"%numPi)
  print("Max Pixel       : %.2e"%max)
  print("Number of Cows  : %d"%count)
  print("Number of Import: %d"%pixs)
  print("Number of Patch : %d"%numPa)
  print("Number of PPix  : %d"%PatPi)
  print("Full Image:")
  print("Count Sparsity  : %.2e"%(count/numPi))
  print("Pixel Sparsity  : %.2e"%(pixs /numPi))
  print("Selected ROI:")
  print("Count Sparsity  : %.2e"%(count/PatPi))
  print("Pixel Sparsity  : %.2e"%(pixs /PatPi))
  print("Ratio:")
  print("Full / Patch    : %.2f"%((numPi)/(PatPi)))


def main():
  img      = 'IMG_1433'
  split    = 'Train'
  base_dir = 'E:/Binalab-Animal-detection-and-counting/Data'
  generator = DataGenerator(split,base_dir)
  get_patches(generator,img,base_dir)


main()
