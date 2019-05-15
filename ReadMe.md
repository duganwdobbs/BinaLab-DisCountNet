## DisCountNet
# Discriminating and Counting Network for Real-Time Counting and Localization of Sparse Objects in High-Resolution UAV Imagery

Welcome to the public repository for DisCountNet. The goal of DisCountNet is to
provide a real-time solution to counting and localization of cows in a natural
setting. To achieve this, we generate assumptions on observations of data, and
then build a model to conform to those assumptions.

<img src="http://therockportgeek.com/DisCountNet.png" width="400">

If you are using the code or theory provided here in a publication, please cite our paper:

@article{rahnemoonfar2019discountnet,
  title={DisCountNet: Discriminating and Counting Network for Real-Time Counting and Localization of Sparse Objects in High-Resolution UAV Imagery},
  author={Rahnemoonfar, Maryam and Dobbs, Dugan and Yari, Masoud and others},
  journal={Remote Sensing},
  volume={11},
  number={9},
  pages={1128},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}


### Training DisCountNet
Training is preformed by running 'python3 Train.py ARGUMENTS'. The arguments can
be set manually internally, or by hand through command line arguments. Please
see the file for more details.

### Testing DisCountNet
The testing script for DisCountNet will pull the most recent model
automatically. To do this, run 'python3 Test.py ARGUMENTS'

Requirements: Tensorflow 1.6 - The Neural Network Framework
              OpenCV and MatPlotLib for Visualization
              Timeit to time the network
              PIL for faster image IO than CV2
              os,random,pickle,json libraries for working with files and lists
              skimage library for comparing SSIM
