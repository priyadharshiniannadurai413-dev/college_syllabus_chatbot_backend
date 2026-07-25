from app.rag.rag_pipeline import get_rag_prompt
from app.ai.fallback_service import FallbackService


class ChatService:


    def __init__(self, user_prompt, is_voice=False):
        self.user_prompt = user_prompt
        self.is_voice = is_voice

    async def chat(self):
        prompt = await get_rag_prompt(
            self.user_prompt,
            is_voice=self.is_voice
        )

        fallback = FallbackService(prompt)

        async for chunk in fallback.chat():
            yield chunk