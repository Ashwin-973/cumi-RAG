from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.vectorstore import get_index, get_pinecone_client

'''
    build a passage string for the re-ranker using metadata stores in each vector
'''
def _build_passage(meta: dict[str, Any]) -> str:

    parts = [
        meta.get("product_name", ""),
        meta.get("brand", ""),
        meta.get("category", ""),
        meta.get("sub_category", ""),
        meta.get("specifications", ""),
    ]
    return " ".join(p for p in parts if p).strip()

'''
    retrieve and re-rank the most relevant products using the search query
'''
def search_products(
    user_query: str,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:

    top_k  = top_k or settings.SEARCH_TOP_K
    pc     = get_pinecone_client()
    index  = get_index()

    #*Stage 1: embed query -> dense vector search
    embed_resp = pc.inference.embed(
        model=settings.PINECONE_EMBED_MODEL,
        inputs=[user_query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    query_vector: list[float] = embed_resp[0]["values"]

    query_kwargs: dict[str, Any] = {
        "vector": query_vector,
        "top_k": settings.SEARCH_TOP_N,
        "include_metadata": True,
    }
    if metadata_filter:
        query_kwargs["filter"] = metadata_filter

    raw = index.query(**query_kwargs)
    candidates: list[dict] = raw.get("matches", []) #*get the search result as a list of dicts

    if not candidates:
        return []

    #*Stage 2: re-rank results with a cross encoder
    passages = [_build_passage(m["metadata"]) for m in candidates]

    rerank_resp = pc.inference.rerank(
        model=settings.PINECONE_RERANK_MODEL,
        query=user_query,
        documents=passages,
        top_n=top_k,
        return_documents=False,
    )

    #*Stage 3: form the result dicts
    results: list[dict[str, Any]] = []
    for item in rerank_resp.data:
        meta = candidates[item.index]["metadata"]
        results.append({
            "rerank_score":   round(item.score, 6),
            "product_name":   meta.get("product_name", ""),
            "brand":          meta.get("brand", ""),
            "category":       meta.get("category", ""),
            "sub_category":   meta.get("sub_category", ""),
            "url":            meta.get("url", ""),
            # "image_url":      meta.get("image_url", ""),
            "catalogue_url":  meta.get("catalogue_url", ""),
            "specifications": meta.get("specifications", ""),
        })

    return results