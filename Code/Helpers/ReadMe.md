This is the helpers directory of the DisCountNet repository.

In this directory, you will find four files:

- filetools.py: These scripts are for working with files and directories, neatly
                wrapped functions to organize, find, and use files.

- generator.py: This is the data generator used for training and testing. It
                pulls the testing / training split from saved files in the Data
                directory.

- ops.py: This directory generally contains wrapped TensorFlow operations.
          They translate lower level code to more default arguments and neater
          calls.

- ulil.py: This file contains CPU or helper operations necessary to run the
           network.
