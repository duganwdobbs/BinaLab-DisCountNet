# DisCountNet
## Discriminating and Counting Network for Real-Time Counting and Localization of Sparse Objects in High-Resolution UAV Imagery

Welcome to the public repository for DisCountNet. The goal of DisCountNet is to
provide a real-time solution to counting and localization of cows in a natural
setting. To achieve this, we generate assumptions on observations of data, and
then build a model to conform to those assumptions.
<p align="center">
<img src="http://therockportgeek.com/DisCountNet.png" width="600">
</p>
Recent deep learning counting techniques revolve around two distinct features of data, sparse data which favors detection networks, or dense data where density map networks are used. Both of these techniques fail to address a third scenario where dense objects are sparsely located. Raw aerial images represent sparse distributions of data in the majority of situations. To address this issue, we propose a novel and exceedingly portable end-to-end model, DisCountNet, and an example dataset to test it on. DisCountNet is a two stage network that utilizes theories from both detection and heat map networks to provide a simple, yet powerful design. The first stage, DiscNet, operates on the theory of coarse detection, but does so by converting a rich and high resolution image into a sparse representation where only important information is encoded. Following this, CountNet operates on the dense regions of the sparse matrix to generate a density map, which provides fine locations and count predictions on densities of objects. Comparing the proposed network to current state-of-the-art networks, we find that we are able to maintain competitive performance while utilizing a fraction of the computational complexity, resulting in a real-time solution.


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

Requirements: 

Tensorflow 1.6 - The Neural Network Framework

OpenCV and MatPlotLib for Visualization

Timeit to time the network

PIL for faster image IO than CV2

os,random,pickle,json libraries for working with files and lists

skimage library for comparing SSIM
