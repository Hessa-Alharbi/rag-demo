from fastapi import Request
from fastapi.responses import JSONResponse
from core.errors import APIError
from loguru import logger

async def error_handler(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        
        if isinstance(e, APIError):
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )
            
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "details": {"error": str(e)}
            }
        )
