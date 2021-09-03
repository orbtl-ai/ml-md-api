import json
import os
import shutil
from typing import List

import fastapi
from fastapi import Depends, File, UploadFile
from fastapi.responses import FileResponse

from data_models.user_submission import User_Submission

from PIL import Image

from api.api_utils.drawing_utils import plot_bboxes
from api.api_utils.inference_utils import batch_inference, load_model
from api.api_utils.preprocessing_utils import (calc_gsd, chip,
                                               dont_resize_to_gsd,
                                               ingest_image, reassemble_chips,
                                               resize_to_gsd)
from api.api_utils.server_utils import clean_temporary_files, security_check
#from api.data_models.user_submission import User_Submission

#HARDCODED API PARAMETERS
TARGET_GSD_CM=2.0
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/tiff"]

# SAVE FILE LOCATIONS
CHIP_IMAGE_PATH="/app_data/chips"
FINAL_OUTPUT_PATH="/app_data/final_outputs"
FINAL_ZIP ="/app_data/api_outputs"

# MODELS
LABEL_MAP_PBTXT = "/app/models/efficientdet-d0/md_labelmap_v6_20210810.pbtxt"
PATH_TO_SAVED_MODEL="/app/models/efficientdet-d0/saved_model"
model = load_model(PATH_TO_SAVED_MODEL)

# lookup table for hardcoded sensor parameters. Order is focal_length_mm, sensor_height_cm, sensor_width_cm
SENSOR_DICT = {'skydio2':[3.7, 0.462196, 0.6166660],
                'phantom4pro':[8.8, 0.88, 1.32]}

router = fastapi.APIRouter()

@router.post('/object-detection/')
async def object_detection(aerial_images: List[UploadFile] = File(...), sub: User_Submission = Depends(User_Submission.as_form)):
    '''
    This endpoint will accept non-georeferenced, 2 centimeter aerial imagery typically taken from airplane or Unmanned Aerial Systems (UAS).\n
    
    Optional image resampling to 2 centimeter resolution can be performed if desired. This requires the user to submit additional flight parameters.\n
    
    The resulting zip file from this operation can be retrieved at the associated /object-detection-results/ GET endpoint.

    INPUTS: 
      -  aerial_images: a list of non-georeferenced aerial images of coastal zones on which marine debris object detection is to be performed
      -  skip_optional_resampling: a boolean (true/false) value that specifies whether to skip optional resampling.
      -  flight_AGL: a decimal (float) value of the aerial sensor's height above ground level when the imagery was collected. This is neccacary to 
           resample input imagery to the API's desired 2 centimeter ground spacing distance (GSD). This value is optional when skip_optional_resampling=True.
      - sensor_platform (optional): a string value indicating the platform used for collection. 'skydio2' and 'phantom4pro' are currently supported. 
           This value is optional when skip_optional_resampling=True.
      - confidence threshold (optional): each prediction from the computer vision model has a confidence score attached. This threshold filters low confidence
           detections from being shown on the image plots. By default this value is set to 0.3 (30% confidence). Recommended values range from 0.2 to 0.5.

    OUTPUTS:
      -  MESSAGE: "Successful Object Detection!"
      -  a compressed file (.zip) which contains:
           1. Image chips showing the location, classification, and confidence score for each predicted marine debris object in the input files.
           2. A JSON file that contains lists of the bboxes, classes, and scores for each predicted marine debris object in each image chip.
    '''
    clean_temporary_files(CHIP_IMAGE_PATH, FINAL_OUTPUT_PATH)

    print(f"Received {len(aerial_images)} images.")
    screened_images = security_check(aerial_images, ALLOWED_CONTENT_TYPES)
    print(f"Accepted {len(screened_images)} images.")
    
    for file in screened_images:
        base_img_name, base_img_ext = os.path.splitext(file.filename)
        chip_base_img_path = os.path.join(CHIP_IMAGE_PATH, file.filename)

        img_content = await file.read()

        in_image = ingest_image(img_content)

        if sub.skip_optional_resampling == True:
          print(f"User declined automatic resampling.")
          processed_image = dont_resize_to_gsd(in_image, chip_base_img_path)
        else:
          print(f"User opted in to automatic resampling.")
          if sub.sensor_platform in SENSOR_DICT.keys():
            sensor_focal_length, sensor_height, sensor_width = SENSOR_DICT[sub.sensor_platform]

            est_gsd_height, est_gsd_width = calc_gsd(sub.flight_AGL, sensor_focal_length, in_image.height, in_image.width, sensor_height, sensor_width)

            max_gsd = max(est_gsd_height, est_gsd_width)
            print(f"Uploaded image's GSD was automatically computed to be {max_gsd} centimeters. Images are going to be resampled to the API's target GSD of {TARGET_GSD_CM} centimeters.")

            processed_image = resize_to_gsd(in_image, max_gsd, chip_base_img_path, TARGET_GSD_CM)
          else:
            print(f"{sub.sensor_platform} is not a supported value. Specify sensor model ('skydio2' or 'phantom4pro') for automatic resampling or value of 'NA' to skip automatic resampling.")
        
        chip_dict = chip(processed_image, base_img_name, base_img_ext, CHIP_IMAGE_PATH)
        
        print("Beginning Inference...")
        inference_results = batch_inference(chip_dict, model, sub.confidence_threshold)

        reassembled_results = reassemble_chips(inference_results)
        
        for k, v in reassembled_results.items():
          plot_bboxes(k, FINAL_OUTPUT_PATH, chip_base_img_path, LABEL_MAP_PBTXT, v, sub.confidence_threshold)

        results_path = os.path.join(FINAL_OUTPUT_PATH, f'{base_img_name}_inference_results.json')
        with open(results_path, 'w') as outfile:
            json.dump(reassembled_results, outfile, indent=0)
      
    shutil.make_archive(FINAL_ZIP, 'zip', FINAL_OUTPUT_PATH)
    clean_temporary_files(CHIP_IMAGE_PATH, FINAL_OUTPUT_PATH)
    return("Successful Object Detection! Retrieve results at /object-detection-results/ endpoint.")

@router.get('/object-detection-results/')
async def receive_results():
    '''
    This GET function returns the zipped prediction results generated by the POST function /object-detection/.

    INPUTS: 
      -  NONE

    OUTPUTS:
      - Latest written zip file containing API object detection predictions.
    '''
    response = FileResponse(FINAL_ZIP+str('.zip'), media_type='application/octet-stream',filename="api_outputs.zip")

    return response
