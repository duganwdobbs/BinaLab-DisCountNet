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

print(imgs)
# input()

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


class LabelWindow():
  def __init__(self,root,image_id=None):
    self.selection = -1
    self.image_id = -1
    self.root = root
    if image_id is None:
      raise KeyError
    self.master = Toplevel(root)
    self.master.title('CowLabeling')
    self.master.attributes('-fullscreen', True)

    # Left Top: Object Type
    # Wrapper frame for left side.
    self.leftFrame = Frame(self.master)
    self.leftFrame.pack(side = LEFT)

    self.buttonFrame = Frame(self.master)
    self.buttonFrame.pack(side = TOP, in_ = self.leftFrame)

    #       Save the data!
    self.saveButton = Button(self.master,text = 'Save Image\nInformation',command = self.save_data)
    self.saveButton.pack(in_ = self.buttonFrame)

    #       Reset (Remove all points for Image)
    self.resetPoints = Button(self.master,text = 'Exit Program',command=exit)
    self.resetPoints.pack(in_ = self.buttonFrame)

    #       Wrapper Frame for Image Selection
    self.selectionFrame = Frame(self.master)
    self.selectionFrame.pack(side = BOTTOM, in_ = self.leftFrame, fill = Y, expand = True)

    #       Frame to have all of the images
    self.imgSelection = Listbox(self.master,height = 40)
    self.imgSelection.pack(in_ = self.selectionFrame, side = LEFT, fill = Y, expand = True)
    [self.imgSelection.insert(x,file_list[x]) for x in range(len(file_list))]

    #       Scroll Bar for our 500,000 images
    self.imgSelectionScroll = Scrollbar(self.master)
    self.imgSelectionScroll.pack(in_ = self.selectionFrame, side = RIGHT, fill = Y)

    self.imgSelection.config(yscrollcommand = self.imgSelectionScroll.set)
    self.imgSelectionScroll.config(command = self.imgSelection.yview)


    # Right: Items in the bin (Can repeat ASIN if multiple of same)
    # One will be active, the active one is the one points are made for.
    self.rightFrame = Frame(self.master)
    self.rightFrame.pack(side = RIGHT)

    self.displayID = Text(self.master,height=1,width=18)
    self.displayID.pack(in_ = self.rightFrame)

    self.displayQuant = Text(self.master,height=1,width=18)
    self.displayQuant.pack(in_ = self.rightFrame)

    self.objectFrame = Frame(self.master)
    self.objectFrame.pack(in_ = self.rightFrame)

    self.itemsList = Listbox(self.master)
    self.itemsList.pack(in_ = self.objectFrame, side = LEFT, fill = Y, expand = True)

    #       Scroll Bar for our items
    self.itemListScroll = Scrollbar(self.master)
    self.itemListScroll.pack(in_ = self.objectFrame, side = RIGHT, fill = Y, expand = True)

    self.itemsList.config(yscrollcommand = self.itemListScroll.set)
    self.itemListScroll.config(command = self.itemsList.yview)

    def next_image():
      idx = imgs.index(self.image_id) + 1
      self.LoadImage(imgs[idx])

    self.nextImageButton = Button(self.master,text = 'Go to\nNext Image',command = next_image)
    self.nextImageButton.pack(in_ = self.rightFrame)

    def prev_image():
      idx = imgs.index(self.image_id) - 1
      self.LoadImage(imgs[idx])

    self.prevImageButton = Button(self.master,text = 'Go to\nPrev. Image',command = prev_image)
    self.prevImageButton.pack(in_ = self.rightFrame)

    def remPoint(event = None):
      self.item.remove_point()
      self.redraw = True

    self.remPointButton = Button(self.master,text = 'Remove\nPoint',command = remPoint)
    self.remPointButton.pack(in_ = self.rightFrame)

    # Bot  : Image frame that can handle OnClick events

    self.imageArea = Label(self.master)

    def addPoint(event):
      self.item.add_point((int(event.x / self.fx),int(event.y / self.fy)))
      self.redraw = True

    def editPoint(event):
      self.item.edit_point((int(event.x / self.fx),int(event.y / self.fy)))
      self.redraw = True

    self.imageArea.bind("<Button-1>",addPoint)
    self.imageArea.bind("<B1-Motion>",editPoint)
    self.imageArea.bind("<ButtonRelease-1>",editPoint)
    self.imageArea.bind("<Button-3>",remPoint)
    self.imageArea.pack(side = BOTTOM,fill = BOTH, expand = 1)

    # self.master.protocol("WM_DELETE_WINDOW", lambda: save_items())
    self.LoadImage(image_id)
    self.update_all()

  def save_data(self):
    to_save = self.item.dump()
    print("SAVING ",to_save)
    with open(os.path.join(json_dir,self.image_id + '.json'),'w') as jsonfile:
      json.dump(to_save,jsonfile)

  def NewImage(self):
    imgSelection = self.imgSelection.curselection()
    if imgSelection is not ():
      imgSelection = imgSelection[0]
      imgSelection = file_list[imgSelection]
      self.LoadImage(imgSelection)


  def LoadImage(self,image_id):
    if image_id is not self.image_id:
      if self.image_id != -1:
        self.save_data()
      self.displayID.delete('1.0',END)
      self.displayID.insert(END,"ID: "+image_id)

      json_file = os.path.join(json_dir,image_id+'.json')

      self.item = ListBuilder(json_file,image_id)

      self.imgSelection.selection_clear(0, END)
      # Save origional Image for later, we can draw on top of it.
      self.image_id = image_id

      # input(os.path.join(img_dir,image_id+'.JPG'))

      self.ori_img = cv2.imread(os.path.join(img_dir,image_id+'.JPG'))
      self.ori_img = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2RGB)
      self.itemsList.delete(0,END)
      [self.itemsList.insert(x,self.item.points[x]) for x in range(len(self.item.points))]

  def DrawSeg(self,selection):
    self.display_img = self.ori_img.copy()

    for point in self.item.points:
      p1,p2 = point
      cv2.rectangle(self.display_img,p1,p2,(0,0,255),4,5)

    self.c_width  = self.imageArea.winfo_width()
    self.c_height = self.imageArea.winfo_height()

    img_h,img_w,img_c = self.display_img.shape
    self.fx    = (self.c_width-2) / img_w
    self.fy    = (self.c_height-2) / img_h

    if self.fx > 0 and self.fy > 0:
      self.display_img = cv2.resize(self.display_img,(0,0),fx = self.fx,fy = self.fy)
      self.photo_img = ImageTk.PhotoImage(image = Image.fromarray(self.display_img))
      self.imageArea.configure(image=self.photo_img)



  def update_all(self):
    self.NewImage()
    selection = self.itemsList.curselection()
    if selection is not self.selection:
      self.redraw = True
      self.selection = selection

    if self.redraw:
      # Redraw Item Information
      if len(selection) > 0:
        self.typevar.set(self.items[selection[0]].type)
      # self.WriteInfo(self.selection)
      self.DrawSeg(self.selection)

      self.itemsList.delete(0,END)
      [self.itemsList.insert(x,self.item.points[x]) for x in range(len(self.item.points))]

    self.master.after(1,func=lambda: self.update_all())


if __name__ == '__main__':
  root = Tk()
  root.wm_withdraw()

  # ourWindow = SelectWindow(root)
  labelWindow = LabelWindow(root,'IMG_1419')

  root.mainloop()
