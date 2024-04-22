from pydantic import BaseModel, Field, EmailStr
from typing import List, Tuple

class PointsSchema(BaseModel):
    points: List[Tuple[float, float]]
    class Config:
        schema_extra = {
            "example": {
                "points": [(10,10),(10,20),(20,10),(20,20)]
            }
        }
class UserSchema(BaseModel):
    fullname: str = Field(...)
    email: EmailStr = Field(...)
    password: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "fullname": "Joe Doe",
                "email": "joe@xyz.com",
                "password": "any"
            }
        }

class UserLoginSchema(BaseModel):
    email: EmailStr = Field(...)
    password: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "email": "joe@xyz.com",
                "password": "any"
            }
        }
