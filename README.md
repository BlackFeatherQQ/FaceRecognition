# Using pytorch,yolocv5+facenet+svm

## Tutorial on CSDN:
https://blog.csdn.net/qq_41334243/article/details/107425492

## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed, including `torch >= 1.5`. To install run:
```bash
$ pip install -U -r requirements.txt
```

## dataset

### *first:
Making database face data,The size of each picture is (160,160),One folder per person<br>
<img src="https://github.com/BlackFeatherQQ/FaceRecognition/blob/master/yolov5_ultralytics/1.JPG" width="500">  
### *then:
Open main function in recognitiuon/test.py,you can see face2database\ClassifyTrainSVC\detect()<br>
The first step is to run face2database<br>
The second step is to run ClassifyTrainSVC<br>
After running, Annotate the two steps above<br>
The third step is to run detect(setOPT()),In the setOPT() method, you can set parameters.

## weights:
### last.pt
[Get it with yolov5 training.](https://github.com/ultralytics/yolov5)<br>
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

<img src="https://github.com/BlackFeatherQQ/FaceRecognition/blob/master/inference/output/0.jpg" width="500">  

## Reproduce Our Training

Download [celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [yolov5](https://github.com/ultralytics/yolov5), install [Apex](https://github.com/NVIDIA/apex) and run command below. I used yolov5s for training,you can use other weights to train your own model.

## quote
yolov5:(https://github.com/ultralytics/yolov5)<br>
blog:(https://blog.csdn.net/ninesky110/article/details/84844307)
