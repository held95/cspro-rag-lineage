from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    container = getattr(request.app.state, "container", None)
    return {
        "status": "ok",
        "ledger": "ok" if container is not None else "starting",
        "model": getattr(getattr(container, "ask", None), "model_name", None),
    }
