### Overview

The code for this project was created by Pierluigi Ferrari in his Github repository [ssd_keras](https://github.com/pierluigiferrari/ssd_keras). The project was copied and adapted for this assignment.

This is a Keras port of the SSD model architecture introduced by Wei Liu et al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

### Dependencies

* Python 3.6
* Numpy
* TensorFlow 2.1.0
* Keras 2.3.1
* OpenCV

Exported environment: environment.yml

### Instructions

 * Install all dependencies from the > environment.yml file (I did not use Jupyter Notebook due to issues with incompatibility of Tensorflow and Jupyter Notebook).
 * (optional) In a case you want to train model on your own dataset:
       * Put your dataset into > datasets folder and set the paths correctly. 
       * Run > preprocess_data.py to preprocess data. All processed pictures will be in a 'train' (there is test data too, we don't use it for training) folder, annotations fill be in a main folder.
       * Run > detection.py to train your model. Settings are for graphic card GTX 1650 Ti (4 GB). I used 5 epochs with 250 steps per epoch. 
* Run > predictions.py to get prediction on the image. You can check on your own picture (first, you need to preprocess that picture with > save_my_pict.py). 
* Run > get_metrics.py to get metrics for the model. I used IoU metric (see > iou.py script for more information). 

### Models

All models and results are saved in a > model_results file.