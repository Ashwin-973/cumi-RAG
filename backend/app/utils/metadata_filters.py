from __future__ import annotations

from typing import Any

#TODO : understand this file
def brand_filter(brand_name: str) -> dict[str, Any]:
    return {"brand": {"$eq": brand_name}}


def category_filter(category: str) -> dict[str, Any]:
    return {"category": {"$eq": category}}

#?wts the exact logic behind this?
def multi_filter(**kwargs: str) -> dict[str, Any]:

    conditions = [
        {field: {"$eq": value}}
        for field, value in kwargs.items()
        if value
    ]
    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}

'''
    match any of the given values for a metadata field (logical OR)
'''
def in_filter(field: str, values: list[str]) -> dict[str, Any]:
    return {field: {"$in": values}}