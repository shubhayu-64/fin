from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

class StockNameModel(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    ticker: str = Field(..., min_length=3, max_length=10)

class StockBaseModel(BaseModel):
    name: str
    price: float
    date: datetime
    is_active: bool = True