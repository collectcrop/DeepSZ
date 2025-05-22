# DeepSZ

Compared to original version, I have modified the original DeepSZ code to make it work with Pytorch. 

## About DeepSZ

DeepSZ [1] is an accuracy-loss expected neural network compression framework, which involves four key steps: network pruning, error bound assessment, optimization for error bound configuration, and compressed model generation, featuring a high compression ratio and low encoding time. The paper is available at: https://dl.acm.org/doi/10.1145/3307681.3326608.

This repo is an implementation of DeepSZ based on Caffe deep learning framework [2] and SZ lossy compressor [3]. Below is the instruction to run DeepSZ on AlexNet using [TACC Frontera system](https://www.tacc.utexas.edu/systems/frontera), which can be adapted to other DNN models (such as VGG-16) and HPC systems with some modifications to the code and scripts (mainly for the network architecture information). 

## Prerequisites
```
Anaconda 3 with Python 3.6+
NVIDIA CUDA 10+
GCC 6.3 or GCC 7.3
pytorch 2.5.1 
SZ 2.0+
tiny-ImageNet-200 validation dataset
```

## Install Pytorch (via Anaconda)
- Download and install Anaconda:
```
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```

- Create conda new environment and install dependencies:
```
conda create -n deepsz_env
conda activate deepsz_env
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pytorch-cuda=12.1 -c pytorch -c nvidia
```

- Download DeepSZ:
```
git clone https://github.com/szcompressor/DeepSZ.git
```

## Download Validation Dataset and DNN Model
- Please download or train AlexNet or other models into DeepSZ /data directory:
here are some dataset links:
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
````

## Run DeepSZ

- After installing PyTorch, please execute **build_SZ.sh** to download and compile SZ lossy compression software.This will be used to compress the network.

- Then, you should train or download the pre-trained model into the DeepSZ /model directory.If you want to use other models, please modify the code to load the model.

- Finally, please use the below command to run DeepSZ to compress the network and test the accuracy with the decompressed model:
```
python deepsz.py --model {model_name}
```
- now it only supports AlexNet, vgg16 and lenet

- Note that the script will automatically download and compile SZ lossy compression software. 

[1] Jin, Sian, Sheng Di, Xin Liang, Jiannan Tian, Dingwen Tao, and Franck Cappello. "Deepsz: A novel framework to compress deep neural networks by using error-bounded lossy compression." In Proceedings of the 28th International Symposium on High-Performance Parallel and Distributed Computing, pp. 159-170. 2019.

[2] Jia, Yangqing, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. "Caffe: Convolutional architecture for fast feature embedding." In Proceedings of the 22nd ACM international conference on Multimedia, pp. 675-678. 2014.

[3] Tao, Dingwen, Sheng Di, Zizhong Chen, and Franck Cappello. "Significantly improving lossy compression for scientific data sets based on multidimensional prediction and error-controlled quantization." In 2017 IEEE International Parallel and Distributed Processing Symposium (IPDPS), pp. 1129-1139. IEEE, 2017. 
