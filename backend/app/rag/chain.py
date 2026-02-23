from __future__ import annotations

import json
from typing import Any

from app.core.llm import generate
from app.rag.prompt import build_rag_prompt
from app.rag.retriever import search_products

'''
    reshape a raw retriever product dict into the RecommendedProduct schema shape
'''
def _shape_product(p: dict[str, Any]) -> dict[str, Any]:

    #*parse specifications JSON string → dict; fall back to empty dict
    try:
        key_specs: dict[str, str] = json.loads(p.get("specifications") or "{}")
    except (json.JSONDecodeError, TypeError):
        key_specs = {}

    return {
        "product_name": p.get("product_name", ""),
        "brand":        p.get("brand", ""),
        "badges":       [p.get("category", ""), p.get("sub_category", "")],
        "key_specs":    key_specs,
        "action_links": {
            "store_url":     p.get("url", ""),
            "catalogue_pdf": p.get("catalogue_url", ""),
        },
    }

'''
    parse the LLM's JSON output; gracefully degrade if it's malformed.
    also handles the nested-JSON case where the LLM stuffs the entire
    response object as a string inside conversational_response.
'''
def _parse_llm_output(raw: str) -> tuple[str, list[str]]:

    try:
        data = json.loads(raw)
        conversational_response = str(data.get("conversational_response", raw))
        suggested_follow_ups    = list(data.get("suggested_follow_ups", []))

        #*Safety net: LLM sometimes nests the full JSON inside conversational_response
        if conversational_response.strip().startswith("{"):
            try:
                nested = json.loads(conversational_response)
                conversational_response = str(nested.get("conversational_response", conversational_response))
                if not suggested_follow_ups:
                    suggested_follow_ups = list(nested.get("suggested_follow_ups", []))
            except (json.JSONDecodeError, AttributeError):
                pass  # not nested JSON — keep as-is

        return conversational_response, suggested_follow_ups
    except (json.JSONDecodeError, AttributeError):
        #*Ollama returned free text instead of JSON — surface it as-is
        return raw, []

'''
    orchestrates the entire retrieve -> prompt -> generate pipeline
    returns a dict matching QueryResponse:
    {
        "conversational_response": str,
        "recommended_products":    list[dict],
        "suggested_follow_ups":    list[str],
    }
'''
def run_rag_chain(
    user_query: str,
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:

    #*STEP 1 — retrieve top-k products from Pinecone
    products = search_products(user_query, top_k=top_k, metadata_filter=metadata_filter)

    #*STEP 2 — build grounded prompt (LLM only generates text + follow-ups)
    prompt = build_rag_prompt(user_query, products)

    #*STEP 3 — call LLM and parse its JSON output
    raw_output = generate(prompt)
    conversational_response, suggested_follow_ups = _parse_llm_output(raw_output)

    #*STEP 4 — assemble recommended_products from retriever data (not LLM)
    recommended_products = [_shape_product(p) for p in products]

    return {
        "conversational_response": conversational_response,
        "recommended_products":    recommended_products,
        "suggested_follow_ups":    suggested_follow_ups,
    }