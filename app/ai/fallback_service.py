import asyncio
import logging

from litellm import acompletion
from litellm.exceptions import RateLimitError, APIError

from app.core.config import settings
from app.prompts.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class FallbackService:

    def __init__(self, user_prompt):

        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

    
        self.fallback_chain = [
            ("Groq", "groq/llama-3.3-70b-versatile", settings.GROQ_API_KEY),
            ("Gemini", "gemini/gemini-2.5-flash", settings.GEMINI_API_KEY)
        ]

    async def stream_with_retry(
        self,
        model_name,
        model,
        api_key,
        retries=3
    ):

        for attempt in range(1, retries + 1):

            try:

                print(f"Trying {model_name} (Attempt {attempt})")

                response = await acompletion(
                    model=model,
                    api_key=api_key,
                    messages=self.messages,
                    temperature=0.7,
                    max_tokens=300,
                    stream=True
                )

                async for chunk in response:

                    content = getattr(
                        chunk.choices[0].delta,
                        "content",
                        None
                    )

                    if content:
                        yield content

                print(f"{model_name} completed successfully")
                return

            except RateLimitError:

                print(f"{model_name} Rate Limited")
                await asyncio.sleep(2)

            except APIError as e:

                print(f"{model_name} API Error : {e}")
                break

            except Exception as e:

                print(f"{model_name} Error : {e}")
                break

        raise Exception(f"{model_name} failed")

    async def chat(self):

        for model_name, model, api_key in self.fallback_chain:

            try:

                print(f"Using Model : {model_name}")  # server-side log only

                async for chunk in self.stream_with_retry(
                    model_name,
                    model,
                    api_key
                ):
                    yield chunk

                return

            except Exception:

                print(f"{model_name} failed. Switching to next model...")

        yield "All models are currently unavailable. Please try again later."