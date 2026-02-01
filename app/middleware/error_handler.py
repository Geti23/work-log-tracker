from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except StarletteHTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "message": e.detail,
                        "status_code": e.status_code,
                        "path": str(request.url.path),
                    }
                },
            )
        except Exception as e:
            logger.error(
                f"Unhandled exception: {e}",
                exc_info=True,
                extra={
                    "path": str(request.url.path),
                    "method": request.method,
                },
            )
            
            error_detail = str(e) if request.app.debug else "Internal server error"
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "message": error_detail,
                        "status_code": 500,
                        "path": str(request.url.path),
                    }
                },
            )

