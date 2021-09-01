from enum import Enum
import fastapi
from starlette.requests import Request
from starlette.templating import Jinja2Templates

templates = Jinja2Templates('templates')
router = fastapi.APIRouter()

class supported_sensors(str, Enum):
  phantom4pro = 'phantom4pro',
  skydio2 = 'skydio2'

@router.get('/')
def index(request: Request):
    return templates.TemplateResponse('home/index.html', context={'request':request, 'choices': [e.value for e in supported_sensors]})

@router.get('/test-api')
async def return_success():
    '''
    A simple function used to test whether the NOAA Machine Learning of Marine Debris backend API is up and running.

    INPUTS:
      -  NONE

    OUTPUTS:
      -  MESSAGE: "The NOAA Machine Learning of Marine Debris API backend server is up and running!"
    '''
    return 'The NOAA Machine Learning of Marine Debris API backend server is up and running!'
