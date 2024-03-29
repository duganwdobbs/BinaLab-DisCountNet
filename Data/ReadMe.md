This is the Data directory, this is where you should put your data. How this is
intended to work is you include all of your images in the Images/ directory,
then use the provided annotation tools in order to generate the metadata. If you
already have your labeled images, you will need to create a new generator, or
feed the data to the network yourself. At the end of the day, DisCountNet is fed
raw high resolution images and probability heat map annotations, then performs
all necessary work internally.

Once images and metadata are in place, this is about how your directory should
look:

- Images/ : The directory to store your data

- metadata/ : The directory to store your metadata

- BBoxWin.py: This is a Bounding Box Annotator that pulls images and metadata
              from their respective folders.

- CountWin.py: This is a center of object point annotator that pulls images and
               metadata from their respective folders.

The previous two files also include Windows batch scripts to automatically run
the tools.

- grid_drawing.py: A visualization tool that draws a grid over an image for
                   ease of viewing patches.

- IMGS.lst: This is a numpy list file that contains the full list of images in
            your data set. NOTE: If you are adding to your data set at any
            point, you need to delete this file and let your annotation tools
            re-parse the directory. They default to this file if it exists for
            speed purposes.

ReadMe.md - This file!

- TEST/TRAIN.lst: This file is generated by the data generator, and ensures that
                  all experiments use the same data split. If you wish to redo
                  this split, delete these files.
