from typing import Optional

from pydantic import BaseModel, validator


class User(BaseModel):
    name: str
    age: int
    email: Optional[str]

    @validator("age")
    def check_age(cls, v):
        if v < 0:
            raise ValueError("age must be non-negative")
        return v

    class Config:
        anystr_strip_whitespace = True


class Order(BaseModel):
    quantity: int
    discount: Optional[float]

    @validator("quantity")
    def check_quantity(cls, v):
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v