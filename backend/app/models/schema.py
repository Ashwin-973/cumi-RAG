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


class ActionLinks(BaseModel):
    store_url:     str
    catalogue_pdf: str
    
''' 
    result of single product returned
'''
class RecommendedProduct(BaseModel):
    product_name: str
    brand:        str
    badges:       list[str]        # [category, sub_category]
    key_specs:    dict[str, str]   # parsed from specifications JSON
    action_links: ActionLinks

'''
    the entire response sent back to the client
'''
class QueryResponse(BaseModel):
    conversational_response: str
    recommended_products:    list[RecommendedProduct]
    suggested_follow_ups:    list[str]