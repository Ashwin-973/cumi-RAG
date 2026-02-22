from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schema import QueryRequest, QueryResponse
from app.rag.chain import run_rag_chain
from app.utils.metadata_filters import multi_filter

router = APIRouter(prefix="/api", tags=["RAG"])

'''
    accepts a query in natural language and returns
        - a grounded LLM generated answer
        - top-k products retrieved
'''
@router.post("/ask", response_model=QueryResponse, summary="Ask the CUMI product AI agent")
async def ask(request: QueryRequest) -> QueryResponse:

    #*Build optional metadata filter from request params
    #?from where does the input for filters come from?
    filters = multi_filter(
        brand=request.brand_filter or "",
        category=request.category_filter or "",
    )

    try:
        result = run_rag_chain(
            user_query=request.query,
            top_k=request.top_k,
            metadata_filter=filters or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc

    #*format response using pydantic
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        products=result["products"],
    )