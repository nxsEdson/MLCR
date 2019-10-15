# MLCR

This is the source code for paper 

"Multi-label Co-regularization for Semi-supervised Facial Action Unit Recognition." </br>
[Xuesong Niu](https://nxsedson.github.io/), Hu Han, Shiguang Shan, Xilin Chen </br>
NeurIPS 2019 </br>

## Highlighted Features

## Database and testing protocol
For EmotioNet database, please refer to [this link](http://cbcsl.ece.ohio-state.edu/dbform_emotionet.html). Please note that we are only able to download 20,722 manually labeled face images. We randomly choose 15,000 images as the labeled training set, and the other manually-labeled images are used for testing. We perform the testing three times and report the average performance.

For BP4D database 

## Pre-processing 

All the faces are detected and aligned using the [SeetaFace Engineer](https://github.com/seetaface/SeetaFaceEngine).

## Environment requirest

This code is based on Python 2.7 and Pytorch 0.4.1.

## Provided model

We provided a model trained on EmotioNet for one testing. You can test it using 'main.py'. The results of this model may be silghtly different from the results in our paper because we reported the average performance of the three testing.

## Things may be useful for you
### loss
### database
### network




