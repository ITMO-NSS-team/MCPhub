from fastapi import APIRouter
from forward.dependencies import ForwardDependencies
from forward.schemas import ForwardRequestDTO, ForwardResponseDTO
import httpx
from fastapi import HTTPException, Request
from config import settings
from const import AskcosEndpoints

router = APIRouter(prefix="/api/v1/forward", tags=["forward"])

@router.post("/predict")
async def forward_predict(
    req: ForwardRequestDTO,
    request: Request,
) -> ForwardResponseDTO:
    """
    Forward prediction of products of a reaction through /api/forward/controller.
    The user sets reactants/reagents/solvent, backend and (optional) model_name.
    """
    client: httpx.AsyncClient = request.app.state.http_client

    body = ForwardDependencies.make_controller_body(req)

    try:
        resp = await client.post(
            settings.ASKCOS_BASE_URL + AskcosEndpoints.FORWARD,
            json=body,
            timeout=60.0,
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"ASKCOS forward error: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ASKCOS forward HTTP {resp.status_code}, response: {resp.text}",
        )

    askcos_json = resp.json()

    try:
        dto = ForwardDependencies.process_forward_result(askcos_json, req)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return dto
