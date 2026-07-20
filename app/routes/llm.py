from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ai.chat_service import ChatService

router = APIRouter()


class ChatRequest(BaseModel):
    user_prompt: str


@router.post("/chatbot")
async def chatbot(request: ChatRequest):
    print(request.user_prompt)

    service = ChatService(request.user_prompt)

    return StreamingResponse(
        service.chat(),
        media_type="text/plain"
    )
    