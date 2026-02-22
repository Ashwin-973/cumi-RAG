from __future__ import annotations

from typing import Any

from app.core.llm import generate
from app.rag.prompt import build_rag_prompt
from app.rag.retriever import search_products

'''
    orchestrates the entire retrieve -> form prompt -> generate response pipeline
    returns a list : {
        "query":    user_query,
        "answer":   answer,
        "products": products,
    } 
'''
def run_rag_chain(
    user_query: str,
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:

    #*STEP 1 — retrieve
    products = search_products(user_query, top_k=top_k, metadata_filter=metadata_filter)

    #*Step 2 — build the grounded rag prompt
    prompt = build_rag_prompt(user_query, products)

    #*Step 3 — geneate response via LLM
    answer = generate(prompt)

    return {
        "query":    user_query,
        "answer":   answer,
        "products": products,
    }