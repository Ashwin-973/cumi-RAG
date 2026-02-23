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

    prompt = f"""You are a product advisor for CUMI Abrasives.

INSTRUCTIONS:
- Answer using ONLY the product context below. Do not invent specs, prices, or names.
- Your ENTIRE output must be a single flat JSON object with exactly two keys.
- No markdown, no code fences, no preamble — raw JSON only.
- "conversational_response" must be a plain text string — NOT a JSON object or nested structure.
- "suggested_follow_ups" must be an array of exactly 3 short strings at the root level.

EXAMPLE OUTPUT (follow this structure exactly):
{{"conversational_response": "Based on your need, the 500 RFT Diamond Segmented Saw is ideal for tile cutting. It has a 110mm diameter and 10 segments, making it compatible with standard angle grinders.", "suggested_follow_ups": ["Do you have this in 125mm?", "What is the price of this blade?", "Can it cut marble as well?"]}}

PRODUCT CONTEXT:
{context_block}

Customer Question: {user_query}

Your JSON output:"""

    return prompt