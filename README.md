# Machine Learning of Marine Debris API (ML/MD API)
## Overview
The Machine Learning of Marine Debris (ML/MD) is a project to automatically detect stranded marine debris objects along coastlines from high resolution aerial photos. Typically, these photographs would be taken from an Uncrewed Aerial System (UAS) or crewed aircraft, allowing rapid standing-stock surveys of marine debris at the local-to-regional scale.

The "automatic detection" of marine debris is performed by deep learning-based object detection models specifically trained for the detection of marine debris. These deep learning models are trained using the [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md).

This entire ML/MD API is "Dockerized", meaning it can be easily downloaded and installed on any personal computer, server, or cloud computing environment. Once the ML/MD API is installed there is no programming or machine learning experience required! The entire project is also open source, meaning it is free to use, share, and modify (forever). 

![An image showing detections of plastic, wood, and other manmade marine debris along a complex shoreline image.](https://github.com/orbtl-ai/md-ml-api/blob/main/images/api_demo_main.png)

## Key Features
- State of the art computer vision models for the automatic detection of stranded shoreline marine debris objects (>20cm) from high-resolution aerial images.
- An Application Programming Interface (API), which allows non-technical users to simply upload imagery and basic flight parameters to receive actionable information about the presence and abundance of marine debris along shorelines.
## Data
Data is not stored in this repo, however there is a labeled data set of approximately 6,000 marine debris objects in 2 centimeter aerial photographs. This data was used to train and evaluate the computer vision models in Tensorflow. This data is available upon request with plans underway to serve the data openly in the future.
## Models
This repo is not designed to host a library of state-of-the-art computer vision models for marine debris. This repo does contain several Tensorflow saved_model.pb files, which are complete Tensorflow programs separated from the original code that built them. A second repo will eventually be stood up to store models associated with this project.
#### faster-rcnn
A Faster R-CNN object detector with an Inception-ResNet-v2 feature classifier. This is currently the best performing combination, however it is a computationally expensive model which takes a long time to run on CPU.
#### EfficientDet-d0
An EfficientDet-d0 object detection model which utilizes a Feature Pyramid Network (FPN) with an EfficientNet feature classifier. This is a well performing combination which balances accuracy with speed of detection. Runs very quickly on CPU.
## Contact
This repo and all associated data, code, models, and documentation was assembled by [ORBTL.AI](ross@orbtl.ai) under funding from NOAA NCCOS and Oregon State University.
# ML/MD API User Guide
## Install the app
This application is installed using [Docker](https://www.docker.com/). Docker allows us to package the entire ML/MD API into a "container" that can be installed in a single command line. Installations will, by default, use the computer's GPU if available. Otherwise all computation will be performed on CPU.

1. Install Docker for your system
2. Clone or copy this repo to your system
3. From the directory, run a ```docker build``` command to buildthe ML/MD API app. We are going to name the app "mdmlapi:latest". 
```
docker build -t mlmdapi:latest .
```
4. Create the following folder structure somewhere on your computer. It doesn't matter where on your computer you make it (just remember where it is, you will need it to run the app)
```
/app-data/
  /chips/
	/final-outputs/
```
## Run the app
One built, the mlmdapi:latest app can be run on your local system using a ```docker run``` command with the **-p** option to open up port 5000, which allows communication between the app and your main computer and the **-v** flag to mount the ```/app-data/``` folder we created earlier, which allows the app to write temporary and output files to this location on your computer.
```
docker run -v /path/to/app_data/:/app_data -p 5000:5000 mdmlapi:latest
```
> **NOTE:** If your system has a Graphics Processing Unit (GPU) then Tensorflow can take advantage of this for speedier computing. Just add the -```-gpus all``` flag to the ```docker run``` command.

If the ML/MD API was installed on a local computer using the port numbers above then the app is most likely accessed by visiting ```localhost:5000/docs/``` in your web browser (or any of the other API endpoints detailed below).

## Use the app
Once running, the app's can be accessed at the following REST API endpoints:
- ```/test-api/``` a POST endpoint that returns an excited, positive affirmation that the ML/MD API app is up and running (if it is, in fact, up and running).
- ```/object-detection/``` a POST endpoint that allows users to upload multiple image files, the type of UAS system, the height above ground level (AGL) the images were taken at.
- ```/object-detection-results/``` A GET endpoint that allows the user to retrieve the latest batch of results from the ML/MD API.

## In-app documentation and interface
Since the entire API is built using [FastAPI](https://fastapi.tiangolo.com/) we are automatically presented with beautful documentation and an interace for testing each API endpoint at the ```/docs/``` endpoint.

![An image showing the API's /docs/ page, which shows additional app info and a testing interface.](https://github.com/orbtl-ai/md-ml-api/blob/main/images/api_docs_v02.png)

## Repo Table of Contents
- /backend - a folder which contains all of the app's backend functionality
  - api_utils.py - utilities used by the API for security and housekeeping
  - drawing_utils.py - utilities used for plotting or returning API results to the user
  - inference.py - utilities used for calling Tensorflow models and performing object detection (inference)
  - process_images.py - utilities used for pre-processing user uploads prior to object detection inference.
- /images - a folder of images displayed in the document you are currently reading!
- server.py - the main app. This file contains all the API configuration and contains the main routine (composed of the various /backend utils).
- Dockerfile - the app's installation routine
- requirements.txt - the app's python dependencies. This is used by Dockerfile during installation.

