from fastapi import FastAPI
import httpx
from config import settings
from router import router as retrosynthesis_router
from forward.router import router as forward_router
from reaction_classification.router import router as reaction_classification_router
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
    base_url=settings.ASKCOS_BASE_URL,
    timeout=settings.HTTP_TIMEOUT,
)
    yield

    await app.state.http_client.aclose()

app = FastAPI(title="Retrosynthesis Proxy API", lifespan=lifespan)


app.include_router(retrosynthesis_router)
app.include_router(forward_router)
app.include_router(reaction_classification_router)




