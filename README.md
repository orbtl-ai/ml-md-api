# Machine Learning of Marine Debris API (ML/MD API)

## Overview

The Machine Learning of Marine Debris (ML/MD) project seeks to automatically detect shoreline stranded marine debris objects from high resolution aerial photos. Typically, these photos are taken from an Uncrewed Aerial System (UAS) or crewed aircraft, allowing a single operator to perform rapid standing-stock surveys of marine debris at the local-to-regional scale.

The "automatic detection" of marine debris is performed by deep learning-based object detection models specifically trained for the detection of marine debris. These deep learning models are trained using the [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md).

This entire ML/MD API is "Dockerized", meaning it can be easily downloaded and installed on any personal computer, server, or cloud computing environment. Once the ML/MD API is installed there is no programming or machine learning experience required! The entire project is also open source, meaning it is free to use, share, and modify (forever).

![An image showing detections of plastic, wood, and other manmade marine debris along a complex shoreline image.](https://github.com/orbtl-ai/md-ml-api/blob/main/images/api_demo_main.png)

## Key Features

- State of the art computer vision models for the automatic detection of stranded shoreline marine debris objects (>20cm) from high-resolution aerial images.
- An Application Programming Interface (API), which allows non-technical users to simply upload imagery and basic flight parameters to receive actionable information about the presence and abundance of marine debris along shorelines.

## Data

Data is not stored in this repo, however there is a labeled data set of approximately 6,000 marine debris objects in 2 centimeter aerial photographs. This data was used to train and evaluate the computer vision models in Tensorflow. This data is available upon request with plans underway to serve the data openly in the future.

## Models

This repo is not designed to host or distribute pre-trained computer vision models for marine debris. However, this repo does contain a ```/models/``` folder with several [Tensorflow saved_model folders](https://www.tensorflow.org/guide/saved_model).

**WARNING: Models provides as-is. No warranty as to accuracy expressed or implied.**

#### faster-rcnn

A Faster R-CNN object detector with an Inception-ResNet-v2 feature classifier. This is currently the best performing combination, however it is a computationally expensive model which takes a long time to run on CPU. It is only recommended to use this model if your hardware has a NVIDIA GPU.

#### EfficientDet-d0

An EfficientDet-d0 object detection model which utilizes a Feature Pyramid Network (FPN) with an EfficientNet feature classifier. This is a well performing combination which balances accuracy with speed of detection. Runs very quickly on CPU.

## Contact

This repo and all associated data, code, models, and documentation are assembled by [ORBTL.AI](ross@orbtl.ai) under funding from NOAA NCCOS and Oregon State University.

# ML/MD API User Guide

## Install the app (Tested on Windows and Linux. Not tested on Mac.)

This application is installed using [Docker](https://www.docker.com/). **Docker allows us to install the entire app in three easy steps.** These steps will work on a personal laptop, a high-powered cloud computing cluster, and anywhere in between!

1. First [install Docker for your system](https://docs.docker.com/engine/install/).
2. Download this repo to your system (green "Code" button in the top right corner) OR [install git for your system](https://git-scm.com/) and clone this repo with the following command:

```
git clone https://github.com/orbtl-ai/md-ml-api.git
```

3. From the repo directory, run a [docker compose](https://docs.docker.com/compose/) command to build, configure, and run the ML/MD API backend server:

```
docker-compose up
```

> **NOTE:** If your system has a Graphics Processing Unit (GPU) this information needs to be passed to the app during the docker compose step. To do this, open the ```docker-compose.yml``` file and remove the leading '#' symbol on lines 8-13 to make those lines active.

## Access the app

If the ML/MD API was installed on a local computer using the port numbers above then the app is most likely accessed by visiting ```localhost:5000/docs/``` in your web browser (or any of the other API endpoints detailed below).

## App Endpoints

- ```/object-detection/``` a POST endpoint that allows users to upload multiple image files, the type of UAS system, the height above ground level (AGL) the images were taken at.
- ```/object-detection-results/``` A GET endpoint that allows the user to retrieve the latest batch of results from the ML/MD API.
- ```/test-api/``` a POST endpoint that returns an excited, positive affirmation that the ML/MD API app is up and running (if it is, in fact, up and running).

## In-app documentation and interface

Since the entire API is built using [FastAPI](https://fastapi.tiangolo.com/) we are automatically presented with beautful documentation and an interace for testing each API endpoint at the ```/docs/``` endpoint.

![An image showing the API's /docs/ page, which shows additional app info and a testing interface.](https://github.com/orbtl-ai/md-ml-api/blob/main/images/api_docs_v02.png)

## Repo Table of Contents

- ```/backend``` - a folder which contains all of the app's backend functionality
  - ```api_utils.py``` - utilities used by the API for security and housekeeping
  - ```drawing_utils.py``` - utilities used for plotting or returning API results to the user
  - ```inference.py``` - utilities used for calling Tensorflow models and performing object detection (inference)
  - ```process_images.py``` - utilities used for pre-processing user uploads prior to object detection inference
- ```/images``` - a folder of images displayed in the document you are currently reading!
- ```server.py``` - the main app. This file contains all the API configuration and contains the main routine (composed of the various ```/backend``` utils)
- ```docker-compose.yml```- the app's installation and configuration routine
- ```Dockerfile``` - the app's build routine
- ```requirements.txt``` - the app's Python dependencies. This is used by ```Dockerfile``` during installation
