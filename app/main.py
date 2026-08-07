"""
app/main.py

Primary entry point for the DE-INTERVIEWER FastAPI application.
Configures CORS, GZip, Request ID, Timing, and Logging middleware,
registers global exception handlers, and includes versioned API routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.constants import API_PREFIX, APP_TITLE, APP_VERSION, APP_DESCRIPTION
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.timing import TimingMiddleware
from app.middleware.logging import APILoggingMiddleware
from app.exceptions.handlers import register_exception_handlers
from app.api.v1.interview import router as interview_router
from app.api.v1.report import router as report_router
from app.api.v1.health import router as health_router

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
)

# ----------------------------------------------------------------------
# Middleware Setup (Execution order is bottom-to-top / outer-to-inner)
# ----------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(APILoggingMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)

# ----------------------------------------------------------------------
# Exception Handler Registration
# ----------------------------------------------------------------------

register_exception_handlers(app)

# ----------------------------------------------------------------------
# Router Inclusion
# ----------------------------------------------------------------------

# Versioned API routes (/api/v1)
app.include_router(interview_router.router, prefix=API_PREFIX)
app.include_router(report_router.router, prefix=API_PREFIX)

# Health & Config routes (top-level /health & /config and /api/v1 aliases)
app.include_router(health_router.router)
app.include_router(health_router.router, prefix=API_PREFIX)
