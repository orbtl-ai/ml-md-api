from PIL import Image
import os
import cv2
import numpy as np
import sys

# TODO: All attempts to add TFODAPI utils to the PYTHONPATH during docker build are failing. This is the only place in the API that
# those are currently used, so inserting a shim until we get PYTHONPATH issues sorted in Docker.

sys.path.insert(0, '/tensorflow/models/research')
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

def category_index(label_map_path) -> dict:
    '''
    Translates a Tensorflow .pbtxt label map file into a Python dictionary that can be used to associate class IDs with class names.
    
    INPUTS:
      -  label_map_path: the path to a Tensorflow .pbtxt label map file.

    OUTPUTS:
      -  category_index: a python dictionary containing a map between class integers and class names 
    '''
    label_map  = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)
    
    category_index = {}
    for i, item in enumerate(categories):
        category_index[i+1] = item
    
    return category_index

def plot_bboxes(output_image_name, output_image_dir, chip_path, label_map_path, detection_dict, CONFIDENCE_THRESHOLD=0.2) -> None:
    '''
    A function that wraps the TFODAPI viz_utils. This can be used to translate our python dictionary of detections into
    a an image chip plot with the model's predictions drawn as bounding boxes with class name and prediction confidence.

    INPUTS:
      -  output_image_name: the desired file name of the output image.
      -  output_image_dir: the desired location of the output image
      -  chip_path: the path to the input image chip to be displayed. Should be an image file.
      -  label_map_path: the path to a Tensorflow .pbtxt label map
      -  detection_dict: A python dictionary that stores the bboxes, classes, and scores for each image chip.
      -  CONFIDENCE_THRESHOLD: A value between 0 and 1.0 that specifies the score threshold at which detections are filtered from our
           plots. This is used to filter low confidence predictions from the data set. Recommended values are between 0.2 and 0.5
           (equivalent to 20% and 50% confidence thresholds).
    
    OUTPUTS:
      -  JPG format images plots are written to the user specified output_image_dir
    '''

    cat_index = category_index(label_map_path)

    bboxes = np.array(detection_dict['bboxes'])
    classes = detection_dict['classes']
    scores = detection_dict['scores']

    image = np.array(Image.open(chip_path))
    image_copy = image.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_copy,
            bboxes,
            classes,
            scores,
            cat_index,
            use_normalized_coordinates=False,
            max_boxes_to_draw=5000,
            min_score_thresh=CONFIDENCE_THRESHOLD,
            agnostic_mode=False,
            line_thickness=1)
    
    output_image_dir = os.path.join(output_image_dir, output_image_name)
    cv2.imwrite(output_image_dir, image_copy)