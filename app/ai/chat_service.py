from app.rag.rag_pipeline import get_rag_prompt
from app.ai.fallback_service import FallbackService

class ChatService:

    def __init__(self, user_prompt):
        self.user_prompt = user_prompt

    async def chat(self):

        prompt = get_rag_prompt(self.user_prompt)
        print("========== FINAL PROMPT ==========")
        print(prompt)
        print("==================================")

        fallback = FallbackService(prompt)

        async for chunk in fallback.chat():
            yield chunk