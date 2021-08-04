import os
import shutil
import json

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from typing import List

from backend.api_utils import security_check, clean_temporary_files
from backend.process_images import ingest_image, calc_gsd, reassemble_chips, resize_to_gsd, chip
from backend.inference import load_model, batch_inference
from backend.drawing_utils import plot_bboxes

# MODELS
LABEL_MAP_PBTXT = "/app/fr_saved_model/dar2015v5_label_map.pbtxt"
PATH_TO_SAVED_FR_MODEL="/app/fr_saved_model"
PATH_TO_SAVED_ED_MODEL="/app/ed_saved_model"

# SAVE FILE LOCATIONS
CHIP_IMAGE_PATH="/app_data/chips"
FINAL_OUTPUT_PATH="/app_data/final_outputs"
FINAL_ZIP ="/app_data/api_outputs"

#HARDCODED API PARAMETERS
TARGET_GSD_CM=2.0
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/tiff"]

# lookup table for hardcoded sensor parameters. Order is focal_length_mm, sensor_height_cm, sensor_width_cm
SENSOR_DICT = {'skydio2':[3.7, 0.462196, 0.6166660],
                'phantom4pro':[8.8, 0.88, 1.32]}

description='''
The Machine Learning of Marine Debris API (ML/MD API) will automatically find large marine debris objects in high-resolution aerial imagery!

Head over to the **/object-detection/** endpoint to start detecting debris with the API.

Retrieve results at the **/object-detection-results/** endpoint after a successful **/object-detection/**.
'''

app = FastAPI(
      title="Machine Learning of Marine Debris API",
      description=description,
      version="0.2",
      contact={
        "name": "ORBTL AI",
        "url": "https://github.com/orbtl-ai/md-ml-api",
        "email": "ross@orbtl.ai",
    },
    license_info={
        "name": "MIT License",
        "url": "https://mit-license.org/",
    },
)

model = load_model(PATH_TO_SAVED_ED_MODEL)

@app.post('/object-detection/')
async def object_detection(aerial_images: List[UploadFile] = File(...), flight_AGL: float = Form(...), 
                            sensor_platform: str = Form(...), confidence_threshold: float = Form(0.2)):
    '''
    This function takes a list of aerial imagery, a user-specified flight above ground level (AGL) value, and a user-specified sensor platform to perform
    object detection with a pretrained object detection model. This routine is not spatially aware. The main output of this method is a zip file containing 
    object detection predictions which can be retrieved at the associated /object-detection-results/ GET endpoint.

    INPUTS: 
      -  aerial_images: a list of non-georeferenced aerial images of coastal zones on which marine debris object detection is to be performed
      -  flight_AGL: a decimal (float) value of the aerial sensor's height above ground level when the imagery was collected. This is neccacary to 
           resample input imagery to the API's desired 2 centimeter ground spacing distance (GSD).
      - sensor_platform: a string value indicating the platform used for collection. 'skydio2' and 'phantom4pro' are currently supported.
      - confidence threshold (optional): each prediction from the computer vision model has a confidence score attached. This threshold filters low confidence
           detections from being shown on the image plots. By default this value is set to 0.2 (20% confidence). Recommended values range from 0.2 to 0.5.

    OUTPUTS:
      -  MESSAGE: "Successful Object Detection!"
      -  a compressed file (.zip) which contains:
           1. Image chips showing the location, classification, and confidence score for each predicted marine debris object in the input files.
           2. A JSON file that contains lists of the bboxes, classes, and scores for each predicted marine debris object in each image chip.
    '''
    clean_temporary_files(CHIP_IMAGE_PATH, FINAL_OUTPUT_PATH)

    print(f"Received {len(aerial_images)} images.")
    screened_images = security_check(aerial_images, ALLOWED_CONTENT_TYPES)
    print(f"Processing {len(screened_images)} accepted image upload(s).")
    
    for file in screened_images:
        base_img_name, base_img_ext = os.path.splitext(file.filename)
        
        img_content = await file.read()

        in_image = ingest_image(img_content)

        if sensor_platform in SENSOR_DICT.keys():
          sensor_focal_length, sensor_height, sensor_width = SENSOR_DICT[sensor_platform]
        else:
          sensor_focal_length, sensor_height, sensor_width = SENSOR_DICT['skydio2']
          print(f"{sensor_platform} is not a supported sensor. Initializing with base with skydio2 parameters.")

        est_gsd_height, est_gsd_width = calc_gsd(flight_AGL, sensor_focal_length, in_image.height, in_image.width, sensor_height, sensor_width)

        max_gsd = max(est_gsd_height, est_gsd_width)
        print(f"Uploaded image's GSD was automatically computed to be {max_gsd} centimeters. Images are going to be resampled to the API's target GSD of {TARGET_GSD_CM} centimeters.")
        
        base_img_path = os.path.join(CHIP_IMAGE_PATH, file.filename)
        processed_image = resize_to_gsd(in_image, max_gsd, base_img_path, TARGET_GSD_CM) 
        
        chip_dict = chip(processed_image, base_img_name, base_img_ext, CHIP_IMAGE_PATH)
        
        print("Beginning Inference...")
        inference_results = batch_inference(chip_dict, model, confidence_threshold)

        reassembled_results = reassemble_chips(inference_results)
        
        for k, v in reassembled_results.items():
          plot_bboxes(k, FINAL_OUTPUT_PATH, base_img_path, LABEL_MAP_PBTXT, v, confidence_threshold)

        results_path = os.path.join(FINAL_OUTPUT_PATH, f'{base_img_name}_inference_results.json')
        with open(results_path, 'w') as outfile:
            json.dump(reassembled_results, outfile, indent=0)
      
    shutil.make_archive(FINAL_ZIP, 'zip', FINAL_OUTPUT_PATH)
    clean_temporary_files(CHIP_IMAGE_PATH, FINAL_OUTPUT_PATH)
    return("Successful Object Detection! Retrieve results at /object-detection-results/ endpoint.")

@app.get('/object-detection-results/')
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

@app.get('/test-api')
async def return_success():
    '''
    A simple function used to test whether the NOAA Machine Learning of Marine Debris backend API is up and running.

    INPUTS:
      -  NONE

    OUTPUTS:
      -  MESSAGE: "The NOAA Machine Learning of Marine Debris API backend server is up and running!"
    '''
    return 'The NOAA Machine Learning of Marine Debris API backend server is up and running!'

if __name__ == "__main__":
    uvicorn.run(app, port='5000', host='0.0.0.0')