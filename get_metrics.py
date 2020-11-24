from keras import backend as K
from keras.models import load_model
import numpy as np
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_data_generator import DataGenerator
from iou import iou

import tensorflow.compat.v1 as tf1
config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf1.InteractiveSession(config=config)


img_height = 360
img_width = 480

normalize_coords = True

val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

images_dir = 'train/'

model_path = 'nadam_custom.h5'

val_labels_filename   = 'annotations_test.csv'


val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
                      include_classes='all')

val_dataset_size   = val_dataset.get_dataset_size()

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session()

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'compute_loss': ssd_loss.compute_loss})

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=False,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)
iou_data = []
no_pred = 0
for j in range(250):
    batch_images, batch_labels, batch_filenames = next(predict_generator)

    y_pred = model.predict(batch_images)

    y_pred_decoded = decode_detections(y_pred,
                                    confidence_thresh=0.3,
                                    iou_threshold=0.3,
                                    top_k=len(batch_labels[0]),
                                    normalize_coords=normalize_coords,
                                    img_height=img_height,
                                    img_width=img_width)

    np.set_printoptions(precision=1, suppress=True, linewidth=90)

    for k in range(len(batch_labels[0])):
        try:
            true_mask = batch_labels[0][k][1:]
            predicted_mask = y_pred_decoded[0][0][2:]
            x_min = true_mask[0]
            y_min = true_mask[1]
            x_max = true_mask[2]
            y_max = true_mask[3]
            rectangle_true = (x_max, y_max, x_max - x_min, y_max - y_min)
            x_min_p = round(predicted_mask[0])
            y_min_p = round(predicted_mask[1])
            x_max_p = round(predicted_mask[2])
            y_max_p = round(predicted_mask[3])

            rectangle_predicted = (x_max_p, y_max_p, x_max_p - x_min_p, y_max_p - y_min_p)
            iou_data.append(iou(rectangle_true, rectangle_predicted))
        except:
            no_pred += 1
print(f'Average IoU is {round(100*  (sum(iou_data) / len(iou_data)))} %. Maximum IoU: {round(100 * max(iou_data), 2)} %. We did not get prediction in {no_pred} / 250 cases')
