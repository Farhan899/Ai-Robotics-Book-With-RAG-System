"""
Middleware for the RAG Chatbot for Robotics Book.
"""
from typing import Callable, Awaitable
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response as StarletteResponse, JSONResponse
from src.core.logging import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> StarletteResponse:
        # Log the incoming request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process the request
        response = await call_next(request)

        # Log the response
        logger.info(f"Response: {response.status_code}")

        return response


class ResponseTimeMiddleware(BaseHTTPMiddleware):
    """
    Middleware for measuring response time.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> StarletteResponse:
        import time

        start_time = time.time()
        response = await call_next(request)
        end_time = time.time()

        response_time = end_time - start_time
        logger.info(f"Request processed in {response_time:.4f}s")

        # Add response time header if not in production
        response.headers["X-Response-Time"] = str(response_time)

        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling errors and exceptions.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> StarletteResponse:
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            # Log the HTTP exception
            logger.error(f"HTTPException: {e.status_code} - {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={"error": f"HTTP_{e.status_code}", "message": str(e.detail)}
            )
        except Exception as e:
            # Log the general exception
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "INTERNAL_SERVER_ERROR", "message": "Internal server error occurred"}
            )


def add_middlewares(app):
    """
    Add all middleware to the FastAPI app.
    """
    app.add_middleware(ResponseTimeMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)