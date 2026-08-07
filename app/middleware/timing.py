"""
app/middleware/timing.py

Middleware measuring execution latency and populating the X-Process-Time header.
"""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.monotonic()
        response = await call_next(request)
        process_time = time.monotonic() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        return response
