import inspect
from typing import Optional, Type
from pydantic import BaseModel, Field, validator
from fastapi import Form

#async def object_detection(aerial_images: List[UploadFile] = File(...), flight_AGL: float = Form(0.0), sensor_platform: str = Form("NA"), confidence_threshold: float = Form(0.2)):
def as_form(cls: Type[BaseModel]):
    """
    Adds an as_form class method to decorated models. The as_form class method
    can be used with FastAPI endpoints

    this function is taken from this solution: https://github.com/tiangolo/fastapi/issues/2387
    """
    new_params = [
        inspect.Parameter(
            field.alias,
            inspect.Parameter.POSITIONAL_ONLY,
            default=(Form(field.default) if not field.required else Form(...)),
        )
        for field in cls.__fields__.values()
    ]

    async def _as_form(**data):
        return cls(**data)

    sig = inspect.signature(_as_form)
    sig = sig.replace(parameters=new_params)
    _as_form.__signature__ = sig
    setattr(cls, "as_form", _as_form)
    return cls

@as_form
class User_Submission(BaseModel):
    skip_optional_resampling: bool = False
    flight_AGL: Optional[float] = Field(None, ge=3.0, le=121.92)
    sensor_platform: Optional[str] = None
    confidence_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0)
    
    @validator('flight_AGL', 'sensor_platform')
    def validate_resampling_settings(cls, v, values):
        if 'skip_optional_resampling' in values and values['skip_optional_resampling'] == False and v == None: 
          raise ValueError('Flight Altitude and Sensor Platform values cannot be left blank if auto-resampling.')
        return v
