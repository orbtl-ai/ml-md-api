# Machine Learning of Marine Debris API (ML/MD API)

## Overview

The Machine Learning of Marine Debris (ML/MD) project seeks to automatically detect shoreline stranded marine debris objects from high resolution aerial photos. Typically, these photos are taken from an Uncrewed Aerial System (UAS) or crewed aircraft, allowing a single operator to perform rapid standing-stock surveys of marine debris at the local-to-regional scale.

The "automatic detection" of marine debris is performed by deep learning-based object detection models specifically trained for the detection of marine debris. These deep learning models are trained using the [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md).

This entire ML/MD API is "Dockerized", meaning it can be easily downloaded and installed on any personal computer, server, or cloud computing environment. Once the ML/MD API is installed there is no programming or machine learning experience required! The entire project is also open source, meaning it is free to use, share, and modify (forever).

![An image showing detections of plastic, wood, and other manmade marine debris along a complex shoreline image.](https://github.com/orbtl-ai/md-ml-api/blob/main/static/api_demo_main.png)

## Key Features

1. State of the art computer vision models for the automatic detection of stranded shoreline marine debris objects (>20cm) from high-resolution aerial images.
2. A RESTful Application Programming Interface (API), which allows technical users to simply upload imagery and basic flight parameters to receive actionable information about the presence and abundance of marine debris along shorelines.

## Data

Data is not stored in this repo, however there is a labeled data set of approximately 6,000 marine debris objects in 2 centimeter aerial photographs. This data was used to train and evaluate the computer vision models in Tensorflow. This data is available upon request with plans underway to serve the data openly in the future.

## Current Models

This repo is not designed to host or distribute pre-trained computer vision models for marine debris. However, this repo does contain a ```/models/``` folder with at least one [Tensorflow saved_model folders](https://www.tensorflow.org/guide/saved_model).

**WARNING: Models provides as-is. No warranty as to accuracy expressed or implied.**

#### EfficientDet-d0

An EfficientDet-d0 object detection model which utilizes a Feature Pyramid Network (FPN) with an EfficientNet feature classifier. This is a well performing combination which balances accuracy with speed of detection. Runs very quickly on CPU.

## Contact

This repo and all associated data, code, models, and documentation are assembled by [ORBTL.AI](ross@orbtl.ai) under funding from NOAA NCCOS and Oregon State University.

# ML/MD API User Guide

## Install the app 

Tested on Windows and Linux. Not tested on Mac, but assumed to work.

This application is downloaded using [git](https://git-scm.com/) and installed using [Docker](https://www.docker.com/). **Docker allows us to install the entire app on any computer hardware and/or operating system in just a few easy steps.** These steps should work on a personal laptop, a high-powered cloud computing cluster, and anywhere in between!

### 1. First [install Docker for your system](https://docs.docker.com/engine/install/) and [install git for your system](https://git-scm.com/)

### 2. Next, download this code repo to your computer by running the following git command line

  ```bash
  git clone https://github.com/orbtl-ai/md-ml-api.git
  ```

### 3. **NOTE: Optional Step for NVIDIA GPU-accelerated hardware:** If your system has a NVIDIA-based Graphics Processing Unit (GPU) this information needs to be passed to the app before running the final installation step below. To pass your GPU information, open the ```docker-compose.yml``` file and remove the leading '#' symbol on lines 8-13 to "activate" those lines and pass your GPU's information to the app

### 4. From the repo directory, run a [docker compose](https://docs.docker.com/compose/) command to build, configure, and run the ML/MD API backend server

  ```bash
  docker-compose up
  ```

You should see a message along the lines of "Successfully built..." if everything went well!

## Access the app's frontend webpage and upload data

If the ML/MD API was installed on a local computer using the port numbers above then the app is most likely accessed by visiting ```localhost:5000/``` in your web browser. If the app is up and running you should see an image similar to below:

![An image showing the API's frontend homepage (/), which has fields for uploading aerial images and flight information to the API.](https://github.com/orbtl-ai/md-ml-api/blob/main/static/api-frontend-beta-v0.3.png)

Once you submit a job, just wait until the API returns the message "Object Detection Successful!". Note that it may take awhile. Final results will be delivered at the /object-detection-results/ endpoint. This is most likely at ```localhost:5000/object-detection-results/``` if you are following along with this install guide. 

## Access the app's backend testing interface and documentation

Since the entire API is built using [FastAPI](https://fastapi.tiangolo.com/) we are automatically presented with beautful documentation and an interface for testing each API endpoint at the ```/docs/``` endpoint. This is most likely at ```localhost:5000/docs``` if you are following along with this install guide.

![An image showing the API's /docs/ page, which shows additional app info and a testing interface.](https://github.com/orbtl-ai/md-ml-api/blob/main/static/api_docs_v02.png)

## App Endpoints

- ```/``` The frontend webpage for uploading aerial imagery to the API.
- ```/object-detection/``` a POST endpoint that allows users to upload multiple image files, the type of UAS system, the height above ground level (AGL) the images were taken at.
- ```/object-detection-results/``` A GET endpoint that allows the user to retrieve the latest batch of results from the ML/MD API.
- ```/test-api/``` a POST endpoint that returns an excited, positive affirmation that the ML/MD API app is up and running (if it is, in fact, up and running).
