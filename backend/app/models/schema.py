"""
backend/app/models/schema.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pydantic v2 request / response schemas for the /ask endpoint.
"""

from __future__ import annotations #* allows new syntax in older python versions

from typing import Any

from pydantic import BaseModel, Field

'''
    incoming client request
'''
class QueryRequest(BaseModel):
    
    query: str = Field(..., min_length=2, max_length=512, description="User's natural-language question.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of products to retrieve.")
    brand_filter:    str | None = Field(default=None, description="Optional: filter by exact brand name.")
    category_filter: str | None = Field(default=None, description="Optional: filter by main category.")

''' 
    result of single product returned
'''
class ProductResult(BaseModel):
    rerank_score:   float
    product_name:   str
    brand:          str
    category:       str
    sub_category:   str
    url:            str
    # image_url:      str
    catalogue_url:  str
    specifications: str

'''
    the entire response sent back to the client
'''
class QueryResponse(BaseModel):
    query:    str
    answer:   str
    products: list[ProductResult]