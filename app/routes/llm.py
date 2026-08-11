from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from app.ai.chat_service import ChatService

router = APIRouter()


class ChatRequest(BaseModel):
    user_prompt: str


@router.post("/chatbot")
async def chatbot(request: ChatRequest):
    # Validate that user_prompt is not empty or whitespace-only
    if not request.user_prompt or not request.user_prompt.strip():
        return JSONResponse(
            status_code=422,
            content={
                "error": "user_prompt cannot be empty.",
                "example": {"user_prompt": "Calculate my CGPA for 8.2, 8.5, 9.0"}
            }
        )

    print(request.user_prompt)

    service = ChatService(request.user_prompt)

    return StreamingResponse(
        service.chat(),
        media_type="text/plain"
    )