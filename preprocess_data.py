import pathlib
from get_coordinates import get_coordinates
from PIL import Image
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np



#prepare test pictures and annotations
pic_list = [p for p in pathlib.Path('datasets/AWEForSegmentation/testannot_rect').iterdir() if p.is_file()]
dataset = dict()
dataset['img_name'] = list()
dataset['x_min'] = list()
dataset['y_min'] = list()
dataset['x_max'] = list()
dataset['y_max'] = list()
dataset['class_name'] = list()
counter = 0

for name in pic_list:
    img_name = str(name)[-8:]
    coordinates, object_n = get_coordinates(str(str(name)[-8:-4]), 'testannot_rect')
    img_dir = 'datasets/AWEForSegmentation/test/' + str(name)[-8:-4] + '.png'
    for j in range(object_n):
        dataset['x_min'].append(coordinates[j][0])
        dataset['y_min'].append(coordinates[j][1])
        dataset['x_max'].append(coordinates[j][2])
        dataset['y_max'].append(coordinates[j][3])
        dataset['img_name'].append(f'ears_test_{counter}.jpg')
        dataset['class_name'].append('1')
    
    
    img = Image.open(img_dir).convert('RGB')
    #img = img.resize((480,300))
    img.save(f'train/ears_test_{counter}.jpg', 'JPEG')
    counter += 1
df_test = pd.DataFrame(dataset)
df_test.to_csv('df_test.csv', index = False, header = None) 


#prepare train pictures and annotations
pic_list_train = [p for p in pathlib.Path('datasets/AWEForSegmentation/trainannot_rect').iterdir() if p.is_file()]
dataset_train = dict()
dataset_train['img_name'] = list()
dataset_train['x_min'] = list()
dataset_train['y_min'] = list()
dataset_train['x_max'] = list()
dataset_train['y_max'] = list()
dataset_train['class_name'] = list()
counter_train = 0

for name in pic_list_train:
    img_name = str(name)[-8:]

    coordinates, object_n = get_coordinates(str(str(name)[-8:-4]), 'trainannot_rect')
    img_dir = 'datasets/AWEForSegmentation/train/' + str(name)[-8:-4] + '.png'

    for j in range(object_n):
        dataset_train['x_min'].append(coordinates[j][0])
        dataset_train['y_min'].append(coordinates[j][1])
        dataset_train['x_max'].append(coordinates[j][2])
        dataset_train['y_max'].append(coordinates[j][3])
        dataset_train['img_name'].append(f'ears_train_{counter_train}.jpg')
        dataset_train['class_name'].append('1')
    img = Image.open(img_dir).convert('RGB')
    #img = img.resize((480,300))
    img.save(f'train/ears_train_{counter_train}.jpg', 'JPEG')
    counter_train += 1
df_train = pd.DataFrame(dataset_train)


ANNOTATIONS_FILE = 'annotations.csv'
CLASSES_FILE = 'classes.csv'

df_train.to_csv(ANNOTATIONS_FILE, index = False, header = ['img_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class'])
df_test.to_csv('annotations_test.csv', index = False, header = ['img_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class'])

classes = set(['ear'])pyth

with open(CLASSES_FILE, 'w') as f:
    for i, line in enumerate(sorted(classes)):
        f.write('{}, {}\n'.format(line, i))