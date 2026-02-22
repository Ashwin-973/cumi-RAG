from __future__ import annotations

from typing import Any

''' 
    construct a grounded prompt[system+user] to be sent to the LLM so it
    - uses only the provided context
    - never hallucinates info on it's own
    - recommend the most relevant product(s) with reasoning
'''
def build_rag_prompt(user_query: str, retrieved_products: list[dict[str, Any]]) -> str:

    if not retrieved_products:
        context_block = "No relevant products were found in the catalogue."
    else:
        sections: list[str] = []
        for i, p in enumerate(retrieved_products, start=1):
            spec_summary = ""
            if p.get("specifications"):
                try:
                    import json
                    specs = json.loads(p["specifications"]) #?in what format are the specifications stored?
                    spec_summary = "; ".join(f"{k}: {v}" for k, v in specs.items() if v)
                except Exception:
                    spec_summary = p["specifications"]

            section = (
                f"Product {i}: {p['product_name']}\n"
                f"  Brand    : {p['brand']}\n"
                f"  Category : {p['category']} > {p['sub_category']}\n"
                f"  URL      : {p['url']}\n"
                f"  Specs    : {spec_summary or 'N/A'}\n"
                f"  Relevance Score: {p['rerank_score']:.4f}"
            )
            sections.append(section)
        context_block = "\n\n".join(sections)

    prompt = f"""You are a knowledgeable product advisor for CUMI Abrasives. \
Answer the customer's question using ONLY the product information provided below. \
Do not invent specifications, prices, or availability. \
If the context does not contain enough information, say so clearly and suggest \
the customer visit the product URL or contact CUMI support.

{context_block} #*PRODUCT CONTEXT

Customer Question: {user_query}

Your Answer (be concise, cite product names, and include the URL when recommending a specific product):"""

    return prompt