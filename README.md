# Using pytorch,yolocv5+facenet+svm

## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed, including `torch >= 1.5`. To install run:
```bash
$ pip install -U -r requirements.txt
```

## Tutorials

* [Notebook](https://github.com/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Google Cloud Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)
* [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)

## dataset

*first:Making database face data,The size of each picture is (160,160),One folder per person
*then:Open main function in recognitiuon/test.py,you can see face2database\ClassifyTrainSVC\detect()<br>
The first step is to run face2database<br>
The second step is to run ClassifyTrainSVC<br>
After running, Annotate the two steps above<br>
The third step is to run detect(setOPT()),In the setOPT() method, you can set parameters.

## weights:
### last.pt
Get it with yolov5 training.<br>
The dataset uses celeba.<br>
You can replace it with your own weight
### 20180402-114759
This is the weight file for facenet.<br>
[rogram and model of downloading facet](https://github.com/davidsandberg/facenet)

## Inference

Inference can be run on most common media formats. 
$ python recognition/test.py --source file.jpg  # image <br>
                             file.mp4  # video<br>
                             ./dir  # directory<br>
                             0  # webcam<br>
                             rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream<br>
                             http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream<br>

<img src="https://user-images.githubusercontent.com/26833433/83082816-59e54880-a039-11ea-8abe-ab90cc1ec4b0.jpeg" width="500">  

## Reproduce Our Training

Download [COCO](https://github.com/ultralytics/yolov5/blob/master/data/get_coco2017.sh), install [Apex](https://github.com/NVIDIA/apex) and run command below. Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                48
                                         yolov5l                                32
                                         yolov5x                                16
```
<img src="https://user-images.githubusercontent.com/26833433/84186698-c4d54d00-aa45-11ea-9bde-c632c1230ccd.png" width="900">

## Reproduce Our Environment

To access an up-to-date working environment (with all dependencies including CUDA/CUDNN, Python and PyTorch preinstalled), consider a:

- **Google Cloud** Deep Learning VM with $300 free credit offer: See our [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Google Colab Notebook** with 12 hours of free GPU time. <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)


## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


## About Us

Ultralytics is a U.S.-based particle physics and AI startup with over 6 years of expertise supporting government, academic and business clients. We offer a wide range of vision AI services, spanning from simple expert advice up to delivery of fully customized, end-to-end production solutions, including:
- **Cloud-based AI** systems operating on **hundreds of HD video streams in realtime.**
- **Edge AI** integrated into custom iOS and Android apps for realtime **30 FPS video inference.**
- **Custom data training**, hyperparameter evolution, and model exportation to any destination.

For business inquiries and professional support requests please visit us at https://www.ultralytics.com. 


## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit https://www.ultralytics.com or email Glenn Jocher at glenn.jocher@ultralytics.com. 
