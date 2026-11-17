"""Pure-ASGI middleware: request id + request body size cap (also for chunked uploads)."""
from __future__ import annotations

import json
import re
import uuid

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse

from app.core.errors import error_body

_RID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class BodyTooLarge(Exception):
    pass


class RequestContextMiddleware:
    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope["headers"]}
        supplied = headers.get(b"x-request-id", b"").decode("latin-1")
        rid = supplied if _RID.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid

        length = headers.get(b"content-length", b"")
        if length.isdigit() and int(length) > self.max_body_bytes:
            await self._too_large(scope, receive, send, rid)
            return

        received = 0
        exceeded = False

        async def limited_receive():
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    exceeded = True
                    raise BodyTooLarge()
            return message

        started = False

        async def send_with_id(message):
            nonlocal started
            if exceeded:
                # FastAPI turns exceptions raised while reading the body into a 400, so once the
                # cap was hit we replace whatever the app answers with a single 413 response.
                if not started:
                    started = True
                    payload = json.dumps(error_body("PAYLOAD_TOO_LARGE", "Request body too large", rid)).encode()
                    await send({"type": "http.response.start", "status": 413, "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(payload)).encode()),
                        (b"x-request-id", rid.encode()),
                    ]})
                    await send({"type": "http.response.body", "body": payload})
                return
            if message["type"] == "http.response.start":
                started = True
                MutableHeaders(scope=message)["X-Request-ID"] = rid
            await send(message)

        try:
            await self.app(scope, limited_receive, send_with_id)
        except BodyTooLarge:
            if started:
                raise
            await self._too_large(scope, receive, send, rid)

    async def _too_large(self, scope, receive, send, rid):
        body = error_body("PAYLOAD_TOO_LARGE", "Request body too large", rid)
        await JSONResponse(body, status_code=413, headers={"X-Request-ID": rid})(scope, receive, send)
