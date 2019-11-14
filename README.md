# MLCR

This is the source code for paper 

[Multi-label Co-regularization for Semi-supervised Facial Action Unit Recognition.](https://arxiv.org/abs/1910.11012) </br>
[Xuesong Niu](https://nxsedson.github.io/), Hu Han, Shiguang Shan, Xilin Chen </br>
NeurIPS 2019 </br>


<img src="./img/pipeline.JPG" width = "600px" height = "250px" align=center />

## Environment requirest

This code is based on Python 2.7, Pytorch 0.4.1 and CUDA 8.0.

## Database and testing protocol
For EmotioNet database, please refer to [this link](http://cbcsl.ece.ohio-state.edu/dbform_emotionet.html). Please note that we are only able to download 20,722 manually-labeled face images. We randomly choose 15,000 images as the labeled training set, and the other manually-labeled images are used for testing. We perform the testing three times and report the average performance. Please refer to our paper for more information.

For BP4D database, please refer to [this link](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html). We conduct a subject-exclusive 3-fold cross-validation. The unlabeled training images used for experiments on BP4D are taken from the EmotioNet database.

## Pre-processing 

All the faces are detected and aligned using the [SeetaFace Engineer](https://github.com/seetaface/SeetaFaceEngine).

## Training

In order to train your model, you need to write your own dataloader. The image transforms used for training is in the 'main.py'. Losses used for training is in the loss file and the usage is in the 'main.py'. More details for training can be found in our paper. 

## Testing

We provided a model trained on EmotioNet for one testing. You can download it from [Google Drive](https://drive.google.com/open?id=1NHXr4UW9eqKqmvPHV6HOAQTPK10KqeE9) or [Baidu Drive](https://pan.baidu.com/s/1PQqV_ConIP5HJG7cYRH3Wg&shfl=sharepset), and test it using 'main.py'. The results of this model may be silghtly different from the results in our paper because we reported the average performance of the three testings. You can use it as a pre-trained model for your task.

## Contact

If you have any problems or any further interesting ideas with this project, feel free to contact me (xuesong.niu@vipl.ict.ac.cn). 

## If you use this work, please cite our paper

    @inproceedings{niu2019multi,
    title={Multi-label Co-regularization for Semi-supervised Facial Action Unit Recognition.},
    author={Niu, Xuesong and Han, Hu and Shan, Shiguang and Chen, Xilin},
    booktitle= {Advances in Neural Information Processing Systems (NeurIPS)},
    year={2019}
    }


