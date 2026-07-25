import sys
from pathlib import Path

# Add project root directory to sys.path to support running as script
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from app.rag.rag_pipeline import get_rag_prompt
from app.ai.chat_service import ChatService
from app.db.mongodb import connect_to_mongo
import asyncio


async def main():

    await connect_to_mongo()

    question = input("Ask a question: ")

    prompt = await get_rag_prompt(question)

    print("\nPrompt sent to the LLM:\n")
    print("=" * 100)
    print(prompt)
    print("=" * 100)

    # Pass the original question so ChatService builds the RAG prompt itself
    service = ChatService(question)

    print("\nAnswer:\n")

    async for chunk in service.chat():
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())