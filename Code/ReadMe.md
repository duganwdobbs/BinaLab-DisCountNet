This is the main code for our experiment for counting cows, DisCountNet. Internally, CountNet is referred to SegNet or SegmNet, as it is a segmentation network

- Train.py (Cow Patch Neural Net)
  This is the MAIN file and script to run. Arguments can be changed to change the experiment.
  TO RUN THE EXPIRMENT, RUN 'python3 Train.py ARGUMENTS'
- Test.py
  This runs a test with the newest saved network state in the default location.

- Helpers
  This directory contains helper files that define implementations used in multiple places.

- discnet.py
  This file defines the discriminator network as a class for ease of use.
- segnet.py
  This file defiens the segmentation network as a class for ease of use.

- Img_Visualization.py
  This script is used to parse an image into patches and display some sparsity statistics about it.


NOTES ON EVALUATION:
You will need to set the base data directory either through command line or default parameters. If you are running it from the base directory of the project, you can run: 'python3 CowPNN.py -base_dir ./Data/'

The code will tell you where logs are saved and the best epoch. Navigate to that epoch's directory, locate the .cmat file, and the do the following:
In order to evaluate metrics, use the metrix.py located in the Code directory. To run it... 'python3 ../Code/metrix.py PATH_TO_CMAT'

NOTE ON FLAGS:
There are four possible experiments set by two flags, Positive Training and Discriminator. They are Boolean flags, I will describe them here:

Positive | Discrimin | Experiment Description
False    | False     | This is the full image training. It doesn't use patches.
True     | False     | This is positive example training, but testing on full images.
True     | True      | This is DiscCount net, learning which patches to not use during testing.
False    | True      | This isn't used, but would be training on all patches, and using the discriminator during testing.
