from fastapi import APIRouter, Depends, HTTPException, status, Request
from httpx import AsyncClient
import httpx
from config import settings
from const import AskcosEndpoints

from .schemas import (
    ReactionClassificationRequestDTO,
    ReactionClassificationResponseDTO,
)
from .dependencies import ReactionClassificationDependencies

router = APIRouter(prefix="/api/v1/reaction-classification", tags=["reaction-classification"])

@router.post(
    "/classify",
    response_model=ReactionClassificationResponseDTO,
    summary="Reaction classification by SMILES (proxy over ASKCOS)",
)
async def classify_reaction(
    req: ReactionClassificationRequestDTO,
    request: Request,
) -> ReactionClassificationResponseDTO:
    """
    Transparent proxy over ASKCOS /api/reaction-classification/call-sync.

    Input: list of reaction SMILES and num_results.  
    Output: same format as from ASKCOS: status_code / message / result[..].
    """
    body = ReactionClassificationDependencies.make_body(req)
    client = request.app.state.http_client
    try:
        resp = await client.post(
            settings.ASKCOS_BASE_URL + AskcosEndpoints.REACTION_CLASSIFICATION,
            json=body,
            timeout=30.0,
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to call ASKCOS reaction-classification: {e}",
        )

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ASKCOS reaction-classification error {resp.status_code}: {resp.text}",
        )

    askcos_json = resp.json()
    dto = ReactionClassificationDependencies.parse_response(askcos_json)
    return dto
