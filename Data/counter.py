from tkinter import *
from PIL import Image, ImageTk
import os,cv2,json,pickle,webbrowser

base_dir = './'
json_dir = base_dir + 'bbox_metadata/'
img_dir  = base_dir + 'Images/'

img_ext  = '.JPG'
json_ext = '_2.json'

code_directory = './'

def file_writer(split,list):
  with open(code_directory + split + '.lst','wb') as fp:
    pickle.dump(list,fp)

def file_reader(split):
  with open(code_directory + split + '.lst','rb') as fp:
    list = pickle.load(fp)
  return list

imgs = []
try:
  imgs = file_reader("IMGS")
  print('LOADED FILE LIST...')
except:
  # input(os.listdir(img_dir))  
  imgs = [f.replace(img_ext,'') for f in os.listdir(img_dir) if f.endswith(img_ext)]
  file_writer("IMGS",imgs)

imgs.sort()
file_list = imgs

print("Files: ",len(imgs))


def ListBuilder(json_file,id):
  try:
    with open(json_file) as f:
      meta   = json.load(f)
  except:
    meta = None
  return ImageLabel(id,meta)

class ImageLabel():
  def __init__(self,id,dump = None):
    print(id)
    print(dump)
    try:
      self.points = dump["points"]
      for x in range(len(self.points)):
        self.points[x] = ((self.points[x][0][0],self.points[x][0][1]),
                          (self.points[x][1][0],self.points[x][1][1]) )
      self.id     = dump["id"]
    except:
        print("Err loading points...")
        self.points = []
        self.id = id

  def add_point(self,point1):
    self.points.append((point1,point1))

  def edit_point(self,point2):
    point1,_ = self.points[-1]
    self.points[-1] = (point1,point2)

  def remove_point(self):
    self.points = self.points[:-1]

  def dump(self):
    dumpstats = {"id":self.id,"points":self.points}
    return dumpstats

sum = 0
for label in imgs:
  sum += len(ListBuilder(json_dir + label + '.json',label).points)
print(sum)
