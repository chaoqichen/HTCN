# Harmonizing Transferability and Discriminability for Adapting Object Detectors 
Implementation of Harmonizing Transferability and Discriminability for Adapting Object Detectors (CVPR 2020)

## Introduction
Please follow [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch) respository to setup the environment. We use **pytorch 1.0.1** and our **CUDA Version is 10.1** for this project. Different versions of pytorch and CUDA will cause errors, which you may find answers to in above link.

### Datasets
* **Cityscape, FoggyCityscape:** Download the website [Cityscape](https://www.cityscapes-dataset.com/), see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data)
* **PASCAL_VOC 07+12:** Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets.
* **Clipart:** Dataset preparation instruction link [Cross Domain Detection](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets).
* **Sim10k:** Website [Sim10k](https://fcav.engin.umich.edu/sim-dataset/)

### Datasets Format
All codes are written to fit for the **format of PASCAL_VOC**.  
If you want to use this code on your own dataset, please arrange the dataset in the format of PASCAL, make dataset class in ```lib/datasets/```, and add it to ```lib/datasets/factory.py```, ```lib/datasets/config_dataset.py```. Then, add the dataset option to ```lib/model/utils/parser_func.py```.

### Data process
Before training, you should use [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to generate the interpolation samples for both source and target domain. And then train the model with original and generated data.  
But if you don't want to bother generating the interpolation samples, just using original data can also achieve competitive results.

## Models
### Pretrained models
We used two pre-trained models on ImageNet as backbone for our experiments, VGG16 and ResNet101. You can download these two models from:
* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and write the path in **__C.VGG_PATH** and **__C.RESNET_PATH** at ```lib/model/utils/config.py```.

### Our trained models
For your convenience, we provide our trained models.
* **Cityscape to FoggyCityscape(VGG16):**
* **PASCAL_VOC to Clipart(ResNet101):**
* **Sim10k to Cityscape(VGG16):**

## Train

## Test

## Citation
Please cite the following reference if you find this repository is helpful.
