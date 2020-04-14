# Harmonizing Transferability and Discriminability for Adapting Object Detectors (CVPR 2020)
A Pytorch Implementation of Harmonizing Transferability and Discriminability for Adapting Object Detectors. 

## Introduction
Please follow [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) respository to setup the environment. In this project, we use Pytorch 1.0.1 and CUDA version is 10.0.130. 

## Datasets
### Datasets Preparation
* **Cityscape and FoggyCityscape:** Download the [Cityscape](https://www.cityscapes-dataset.com/) dataset, see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data).
* **PASCAL_VOC 07+12:** Please follow the [instruction](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC dataset.
* **Clipart:** Please follow the [instruction](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets) to prepare Clipart dataset.
* **Sim10k:** Download the dataset from this [website](https://fcav.engin.umich.edu/sim-dataset/).  

### Datasets Format
All codes are written to fit for the **format of PASCAL_VOC**.  
If you want to use this code on your own dataset, please arrange the dataset in the format of PASCAL, make dataset class in ```lib/datasets/```, and add it to ```lib/datasets/factory.py```, ```lib/datasets/config_dataset.py```. Then, add the dataset option to ```lib/model/utils/parser_func.py```.

### Data Interpolation
You should use [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to generate the interpolation samples for both source and target domain, and then train the model with original and generated data. This step is important for the adaptation from Cityscape to FoggyCityscape and the adaptation from Sim10k to Cityscape. For the adaptation from PASCAL_VOC 07+12 to Clipart, we empirically found that we can also achieve competitive results without interpolation.

## Models
### Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e., VGG16 and ResNet101. Please download these two models from:
* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and write the path in **__C.VGG_PATH** and **__C.RESNET_PATH** at ```lib/model/utils/config.py```.

## Train
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python trainval_net_HTCN.py \
       --dataset source_dataset --dataset_t target_dataset \
       --net vgg16/resnet101 
```
## Test
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python test_net_HTCN.py \
       --dataset source_dataset --dataset_t target_dataset \
       --net vgg16/resnet101  \
       --load_name path_to_model
```
## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{chen2020htcn,
  title={Harmonizing Transferability and Discriminability for Adapting Object Detectors},
  author={Chen, Chaoqi and Zheng, Zebiao and Ding, Xinghao and Huang, Yue and Dou, Qi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
```
@article{chen2020harmonizing,
  title={Harmonizing Transferability and Discriminability for Adapting Object Detectors},
  author={Chen, Chaoqi and Zheng, Zebiao and Ding, Xinghao and Huang, Yue and Dou, Qi},
  journal={arXiv preprint arXiv:2003.06297},
  year={2020}
}
```
