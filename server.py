import os
import shutil
import json
from exif import Image as exImage

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from typing import List

from backend.process_images import ingest_image, calc_gsd, resize_to_gsd, chip, security_check
from backend.inference import load_model, batch_inference
from backend.drawing_utils import plot_bboxes

LABEL_MAP_PBTXT = "/app/fr_saved_model/dar2015v5_label_map.pbtxt"
PATH_TO_SAVED_FR_MODEL="/app/fr_saved_model"
PATH_TO_SAVED_ED_MODEL="/app/ed_saved_model"
CHIP_IMAGE_PATH="/app_data/chips"
FINAL_OUTPUT_PATH="/app_data/final_outputs"
FINAL_ZIP ="/app_data/final_outputs"

CONFIDENCE_THRESHOLD = 0.2
FOCAL_LENGTH_MM_HARDCODE=3.7
TARGET_GSD_CM=2.0

allowed_content_types = ["image/jpeg", "image/png", "image/tiff"]

app = FastAPI()
model = load_model(PATH_TO_SAVED_ED_MODEL)
temp_db = []

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

@app.post('/object-detection/')
async def object_detection(aerial_images: List[UploadFile] = File(...), flight_AGL: float = Form(...)):
    '''
    This function takes a list of aerial imagery and a user-specified flight above ground level (AGL) value to perform
    object detection with a pretrained object detection model (Faster R-CNN with Inception-ResNet-v2). This routine is not
    spatially aware. The main output of this method is a zip file containing object detection predictions which can be retrieved at the
    associated /object-detection-results/ GET endpoint.

    INPUTS: 
      -  aerial_images: a list of non-georeferenced aerial images of coastal zones on which marine debris object detection is to be performed
      -  flight_AGL: a decimal (float) value of the aerial sensor's height above ground level when the imagery was collected. This is neccacary to 
           resample input imagery to the API's desired 2 centimeter ground spacing distance (GSD).

    OUTPUTS:
      -  MESSAGE: "Successful Object Detection!"
      -  a compressed file (.zip) which contains:
           1. Image chips showing the location, classification, and confidence score for each predicted marine debris object in the input files.
           2. A JSON file that contains lists of the bboxes, classes, and scores for each predicted marine debris object in each image chip.
    '''

    print(f"Received {len(aerial_images)} images.")
    screened_images = security_check(aerial_images, allowed_content_types)
    print(f"Processing {len(screened_images)} accepted image upload(s).")
    
    for file in screened_images:
        base_img_name, base_img_ext = os.path.splitext(file.filename)

        img_content = await file.read()
        temp_db.append(img_content)

        in_image = ingest_image(img_content)
        exif_keys = exImage(img_content)

        if exif_keys.focal_length:
            print(f'Reading focal length value of {exif_keys.focal_length} from EXIF keys.')
            est_gsd_height, est_gsd_width = calc_gsd(flight_AGL, exif_keys.focal_length, in_image.height, in_image.width)
        else:
            print(f'No focal length value found in EXIF keys. Using hardcoded focal length value of {FOCAL_LENGTH_MM_HARDCODE}.')
            est_gsd_height, est_gsd_width = calc_gsd(flight_AGL, FOCAL_LENGTH_MM_HARDCODE, in_image.height, in_image.width)
        
        max_gsd = max(est_gsd_height, est_gsd_width)
        if max_gsd != TARGET_GSD_CM:
            print(f"Resizing images from estimated GSD of {max_gsd} to target GSD of {TARGET_GSD_CM}")
            processed_image = resize_to_gsd(in_image, max_gsd, TARGET_GSD_CM) 
        else:
            processed_image = in_image
        
        chip_dict = chip(processed_image, base_img_name, base_img_ext, CHIP_IMAGE_PATH)
        
        print("Beginning Inference...")
        inference_results = batch_inference(chip_dict, model, CONFIDENCE_THRESHOLD)
        
        for k,v in inference_results.items():
            plot_bboxes(k, FINAL_OUTPUT_PATH, os.path.join(CHIP_IMAGE_PATH, k), LABEL_MAP_PBTXT, v, 0.2)
    
    results_path = os.path.join(FINAL_OUTPUT_PATH, 'inference_results.json')
    with open(results_path, 'w') as outfile:
        json.dump(inference_results, outfile, indent=1)

    shutil.make_archive(FINAL_ZIP, 'zip', FINAL_OUTPUT_PATH)
    
    for root, dirs, files in os.walk(CHIP_IMAGE_PATH):
      for file in files:
        os.remove(os.path.join(root, file))

    for root, dirs, files in os.walk(FINAL_OUTPUT_PATH):
      for file in files:
        os.remove(os.path.join(root, file))

    return("Successful Object Detection!")

@app.get('/object-detection-results/')
async def receive_results():
    '''
    This GET function returns the zipped prediction results generated by the POST function /object-detection/.

    INPUTS: 
      -  NONE

    OUTPUTS:
      - Zip file containing API object detection predictions.
    '''
    response = FileResponse(FINAL_ZIP+str('.zip'), media_type='application/octet-stream',filename="api_outputs.zip")

    return response

if __name__ == "__main__":
    uvicorn.run(app, port='5000', host='0.0.0.0')