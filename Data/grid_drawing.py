import cv2
import sys

def draw_grid(file,every_n,color=(255,0,0)):
  img = cv2.imread(file)
  img = cv2.resize(img,(2048,1536))

  img[every_n-1:-1:every_n] = color
  img[:,every_n-1:-1:every_n] = color

  img[every_n::every_n] = color
  img[:,every_n::every_n] = color

  img[every_n+1::every_n] = color
  img[:,every_n+1::every_n] = color

  print("WRITING TO ",file.split('.')[0]+'_GRID.png')
  cv2.imwrite(file.split('.')[0]+'_GRID.png',img)
  return img

if __name__ =='__main__':
  file = sys.argv[1]
  every_n = sys.argv[2]
  draw_grid(file,int(every_n))
