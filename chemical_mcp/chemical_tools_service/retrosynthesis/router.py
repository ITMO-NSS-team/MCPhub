from fastapi import Request, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from schemas import TreeSearchRequest, RetroResponseDTO
from retro_presets import FAST_TREE_SEARCH_BODY, BALANCED_TREE_SEARCH_BODY, DEEP_TREE_SEARCH_BODY
from copy import deepcopy
from uuid import uuid4
from typing import Literal
from dependencies import RetroSyntehsisDependencies
from config import settings
from const import AskcosEndpoints
import httpx

router = APIRouter(prefix="/api/v1/retrosynthesis", tags=["retrosynthesis"])

@router.post("/result")
async def retrosynthesis_result(req: TreeSearchRequest, request: Request, mode: Literal["fast", "balanced", "deep"] = "fast") -> RetroResponseDTO:
    
    PRESETS = {
    "fast": FAST_TREE_SEARCH_BODY,
    "balanced": BALANCED_TREE_SEARCH_BODY,
    "deep": DEEP_TREE_SEARCH_BODY,
    }   
    
    if mode not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")
    
    body = deepcopy(PRESETS[mode])

    body["smiles"] = req.smiles

    # generate a unique result_id to avoid sharing it between requests
    body["result_id"] = str(uuid4())

    try:
        client = request.app.state.http_client
        upstream_resp = await client.post(
                settings.ASKCOS_BASE_URL + AskcosEndpoints.TREE_SEARCH,
                json=body,
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"ASKCOS tree search error: {e}")
    
    result = await RetroSyntehsisDependencies.process_retrosynthesis_result(upstream_resp.json())

    return JSONResponse(
        content=result,
        status_code=upstream_resp.status_code,
    )
