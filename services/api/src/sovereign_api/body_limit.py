"""A small ASGI request-body limit for the generation endpoint."""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_GENERATE_BODY_BYTES = 256 * 1024


class GenerateBodyLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"] == "/v1/generate"
        ):
            await self._app(scope, receive, send)
            return

        content_length = dict(scope["headers"]).get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = MAX_GENERATE_BODY_BYTES + 1
            if declared_size < 0 or declared_size > MAX_GENERATE_BODY_BYTES:
                await self._reject(scope, receive, send)
                return

        messages: list[Message] = []
        received_size = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                messages.append(message)
                break

            received_size += len(message.get("body", b""))
            if received_size > MAX_GENERATE_BODY_BYTES:
                await self._reject(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay_receive() -> Message:
            if messages:
                return messages.pop(0)
            return {"type": "http.disconnect"}

        await self._app(scope, replay_receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "error": {
                    "code": "request_body_too_large",
                    "message": "Request body exceeds the 262144-byte limit",
                }
            },
        )
        await response(scope, receive, send)
