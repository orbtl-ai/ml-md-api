import uvicorn
from fastapi import FastAPI

from api.api_utils.server_utils import create_temp_folders
from api.api_utils.inference_utils import load_model

from views import home
from api import object_detection

# SAVE FILE LOCATIONS
CHIP_IMAGE_PATH="/app_data/chips"
FINAL_OUTPUT_PATH="/app_data/final_outputs"
FINAL_ZIP ="/app_data/api_outputs"

description='''
The Machine Learning of Marine Debris API (ML/MD API) will automatically find large marine debris objects in high-resolution aerial imagery!

Head over to the **/object-detection/** endpoint to start detecting debris with the API.

Retrieve results at the **/object-detection-results/** endpoint after a successful **/object-detection/**.
'''

api = FastAPI(
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

def configure():
  '''
  A simple function that configures the API on startup. Currently just configures routing, but built to handle more in the future.
  '''
  configure_routing()

def configure_routing():
  '''
  Configures the API's routers
  '''
  #api.mount('/static', StaticFiles(directory='static'), name='static')
  api.include_router(home.router)
  api.include_router(object_detection.router)


create_temp_folders(CHIP_IMAGE_PATH, FINAL_OUTPUT_PATH)


if __name__ == "__main__":
  configure()
  uvicorn.run(api, port='5000', host='0.0.0.0')
else:
  configure()