from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()                                      

PINECONE_API_KEY: str = os.environ["PINECODE_DEFAULT_API_KEY"]

INDEX_NAME      = "cumi-abrasives-index"
EMBED_MODEL     = "llama-text-embed-v2"      
RERANK_MODEL    = "bge-reranker-v2-m3"            
DIMENSION       = 1024
METRIC          = "cosine"
UPSERT_BATCH    = 96                                
SEARCH_TOP_N    = 30                               
EXCEL_PATH      = Path(__file__).parents[1] / "data//processed//products.xlsx"

pc = Pinecone(api_key=PINECONE_API_KEY)


#* STAGE 1 : data preparation

'''
    convert the raw specifications cell into formatted text
    example output: "Type: Cutting Disc; Diameter: 230 mm; Thickness: 2.0 mm"
'''
def _parse_specifications(raw: Any) -> str:

    if pd.isna(raw) or raw == "":
        return ""
    try:
        specs: dict = json.loads(raw) if isinstance(raw, str) else raw
        return "; ".join(f"{k}: {v}" for k, v in specs.items() if v)
    except (json.JSONDecodeError, TypeError):
        return str(raw).strip()

'''
    construct  a golden string that contains every semantic meaning of a vector worth capturing
    example format : "{Name} is a {sub_cat} product by {brand}, in the {main_cat} category.
         Specifications: {spec_summary}."
'''
def _build_golden_string(row: pd.Series) -> str:

    name      = row.get("Product Name", "") or ""
    brand     = row.get("Brand Name", "")   or ""
    main_cat  = row.get("Main Category", "") or ""
    sub_cat   = row.get("Sub Category", "")  or ""
    spec_str  = _parse_specifications(row.get("Specifications", ""))

    parts: list[str] = []

    # core identity sentence
    if name:
        identity = f"{name}"
        if sub_cat:
            identity += f" is a {sub_cat} product"
        if brand:
            identity += f" by {brand}"
        if main_cat:
            identity += f", in the {main_cat} category"
        parts.append(identity + ".")

    # specification detail sentence
    if spec_str:
        parts.append(f"Specifications: {spec_str}.")

    return " ".join(parts) if parts else name

'''
    load the excel file , standardize data
    prepare the df with "text_embed" [golden string] column for embedding
'''
def load_and_prepare(excel_path: Path = EXCEL_PATH) -> pd.DataFrame:

    print(f"[DATA] Loading Excel from: {excel_path}")
    df = pd.read_excel(excel_path, engine="openpyxl")

    # Normalise column names (strip accidental whitespace)
    df.columns = df.columns.str.strip()

    # Fill NaN with empty strings for string columns; keep others as-is
    str_cols = ["Product Name", "Brand Name", "Main Category", "Sub Category",
                "Specifications", "Product URL", "Catalogue PDF URL"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Drop rows that have neither a name nor a URL (completely empty)
    df = df[~((df["Product Name"] == "") & (df.get("Product URL", pd.Series(dtype=str)) == ""))]
    df = df.reset_index(drop=True)

    df["text_to_embed"] = df.apply(_build_golden_string, axis=1)

    print(f"[DATA] Prepared {len(df)} products for ingestion.")
    return df


#* STAGE 2 : playing with pinecone index
'''
    creates serveless index if it doesn't already exist
    polls repeatedly until index is ready
    returns pc.index(INDEX_NAME)
'''
def get_or_create_index() -> Any:

    existing = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"[INDEX] Creating serverless index '{INDEX_NAME}' …")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Poll until the index is ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("[INDEX] Waiting for index to become ready …")
            time.sleep(3)
        print("[INDEX] Index is ready.")
    else:
        print(f"[INDEX] Index '{INDEX_NAME}' already exists — skipping creation.")

    return pc.Index(INDEX_NAME)


#* STAGE 3 : batch embedding and upsert
'''
    construct metadata to be stored along each each vector
'''
#?are we storing all context?
def _build_metadata(row: pd.Series) -> dict[str, Any]:

    return {
        "product_name":  row.get("Product Name", ""),
        "brand":         row.get("Brand Name", ""),
        "category":      row.get("Main Category", ""),
        "sub_category":  row.get("Sub Category", ""),
        "url":           row.get("Product URL", ""),
        "catalogue_url": row.get("Catalogue PDF URL", ""),
        "specifications": row.get("Specifications", ""),  
    }

'''
    embed and upsert df["text_to_embed"]
    each vector has :
        id → UUID v4
        values → 1024-dim float
        metadata
'''
def ingest(df: pd.DataFrame, index: Any) -> None:

    texts  = df["text_to_embed"].tolist()
    total  = len(texts)
    print(f"[INGEST] Starting ingestion of {total} products in batches of {UPSERT_BATCH} …")

    for start in tqdm(range(0, total, UPSERT_BATCH), desc="Upserting batches"):
        batch_df    = df.iloc[start : start + UPSERT_BATCH]
        batch_texts = texts[start : start + UPSERT_BATCH]

        # Generate embeddings via Pinecone Inference API
        embed_response = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=batch_texts,
            parameters={"input_type": "passage", "truncate": "END"},
        )

        # Build the upsert payload
        vectors = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            vectors.append({
                "id":       str(uuid.uuid4()),
                "values":   embed_response[i]["values"],
                "metadata": _build_metadata(row),
            })

        index.upsert(vectors=vectors)

    print(f"[INGEST] ✓ {total} products successfully upserted to '{INDEX_NAME}'.")

'''
    orchestrate the data ingestion pipeline load -> prepate -> create index -> embed -> upsert
'''
def run_ingestion() -> None:
    df    = load_and_prepare()
    index = get_or_create_index()
    ingest(df, index)


#*STAGE 4 : smart search (dense search + re-renking)
def search_products(
    user_query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Two-stage semantic retrieval with reranking.

    Stage A — Vector Search:
        Embed the query with Llama and fetch the top-SEARCH_TOP_N candidates
        from Pinecone (wide net to maximise recall before reranking).

    Stage B — Reranking:
        Pass all candidates + the raw query text to the BGE reranker.
        The reranker scores by cross-attention (much more precise than cosine).

    Stage C — Return:
        Slice the top `top_k` reranked results as a clean list of dicts.

    Parameters
    ----------
    user_query : str
        Natural language product search query.
    top_k : int
        Number of final results to return after reranking.

    Returns
    -------
    list[dict]
        Each dict contains: score, product_name, brand, category,
        url, catalogue_url, specifications.
    """
    index = pc.Index(INDEX_NAME)

    #*dense vector search
    query_embed = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[user_query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    query_vector: list[float] = query_embed[0]["values"]

    search_results = index.query(
        vector=query_vector,
        top_k=SEARCH_TOP_N,
        include_metadata=True,
    )
    candidates = search_results.get("matches", [])

    if not candidates:
        return []

    #*cross-encoder reranking
    # Build a list of text passages for the reranker from stored metadata
    passages = [
        (
            f"{m['metadata'].get('product_name', '')} "
            f"{m['metadata'].get('brand', '')} "
            f"{m['metadata'].get('category', '')} "
            f"{m['metadata'].get('sub_category', '')} "
            f"{m['metadata'].get('specifications', '')}"
        ).strip()
        for m in candidates
    ]

    rerank_response = pc.inference.rerank(
        model=RERANK_MODEL,
        query=user_query,
        documents=passages,
        top_n=top_k,
        return_documents=False,          # we keep original metadata; no need to echo docs
    )

    #*qssemble final results
    results: list[dict[str, Any]] = []
    for item in rerank_response.data:
        original_idx = item.index                # maps back to `candidates`
        meta = candidates[original_idx]["metadata"]
        results.append({
            "rerank_score":  round(item.score, 6),
            "product_name":  meta.get("product_name", ""),
            "brand":         meta.get("brand", ""),
            "category":      meta.get("category", ""),
            "sub_category":  meta.get("sub_category", ""),
            "url":           meta.get("url", ""),
            # "image_url":     meta.get("image_url", ""),
            "catalogue_url": meta.get("catalogue_url", ""),
            "specifications": meta.get("specifications", ""),
        })

    return results


#*the entry point

if __name__ == "__main__":
    # Step 1: Ingestion
    # Run ONCE to populate the index, then comment out to avoid duplicate upserts.
    # run_ingestion()

    # Step 2: Test Smart Search
    TEST_QUERY = "grinding centerless"
    print(f"\n[SEARCH] Query: '{TEST_QUERY}'\n{'─' * 60}")

    hits = search_products(TEST_QUERY, top_k=5)

    if not hits:
        print("[SEARCH] No results found.")
    else:
        for rank, hit in enumerate(hits, start=1):
            print(
                f"  #{rank}  [{hit['rerank_score']:.4f}]  "
                f"{hit['product_name']}  |  {hit['brand']}  |  {hit['category']}"
            )
            print(f"       URL: {hit['url']}")
            if hit.get("specifications"):
                # Pretty-print first 120 chars of the specs JSON
                print(f"       Specs: {hit['specifications'][:120]} …")
            print()