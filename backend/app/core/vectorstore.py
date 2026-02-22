from __future__ import annotations

from functools import lru_cache
from typing import Any

from pinecone import Pinecone

from app.core.config import settings

'''
    return a module-level cached pinecone client
'''
@lru_cache(maxsize=1) #?tf does this do?
def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=settings.PINECONE_API_KEY)

'''
    return a cached handle to the product index
'''
@lru_cache(maxsize=1)
def get_index() -> Any:
    pc = get_pinecone_client()
    return pc.Index(settings.PINECONE_INDEX_NAME)