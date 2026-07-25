import os
from fastapi import APIRouter, UploadFile, File
import uuid
import time



from app.services.speech_service import (
    speech_to_text,
    text_to_speech
)
from app.ai.chat_service import ChatService

router = APIRouter(
    prefix="/voice",
    tags=["Voice"]
)

@router.post("/chat")
async def voice_chat(file: UploadFile = File(...)):
    total_start = time.perf_counter()
    # 1. Save uploaded audio to uploads/
    save_start = time.perf_counter()
    os.makedirs("uploads", exist_ok=True)
    audio_path = os.path.join("uploads", file.filename)

    with open(audio_path, "wb") as f:
        f.write(await file.read())
    
    print(f"Audio Save Time: {time.perf_counter() - save_start:.2f}s")
    # 2. Call speech_to_text(audio_path)
    stt_start = time.perf_counter()
    transcribed_text = await speech_to_text(audio_path)
    print(f"STT Time: {time.perf_counter() - stt_start:.2f}s")

    

    voice_prompt = f"""
You are a college syllabus assistant.

Answer briefly and clearly.
Use simple language.
Keep the answer within 2-3 sentences.
Do not use markdown or bullet points unless necessary.

User question:
{transcribed_text}
"""

    # 3. Pass transcribed text to existing ChatService & get RAG pipeline response
    chat_start = time.perf_counter()
    chat_service = ChatService(
    transcribed_text,
    is_voice=True
)
    print(f"ChatService Init: {time.perf_counter() - chat_start:.2f}s")

    llm_start = time.perf_counter()
    response_chunks = []
    async for chunk in chat_service.chat():
        response_chunks.append(chunk)
    chatbot_response = "".join(response_chunks)
    print(f"RAG + LLM: {time.perf_counter() - llm_start:.2f}s")

    # 4. Convert chatbot response using text_to_speech and save MP3 to outputs/
    os.makedirs("outputs", exist_ok=True)
    output_filename = f"{uuid.uuid4()}.mp3"
    output_audio_path = os.path.join("outputs", output_filename)

    tts_start = time.perf_counter()
    await text_to_speech(chatbot_response, output_audio_path)
    print(f"TTS Time: {time.perf_counter() - tts_start:.2f}s")


    print(f"Total Time: {time.perf_counter() - total_start:.2f}s")

    # 5. Return response details
    return_start = time.perf_counter()
    return {
        "transcribed_text": transcribed_text,
        "chatbot_response": chatbot_response,
        "audio_url": f"http://localhost:8000/outputs/{output_filename}"
    }
    print(f"Response Build: {time.perf_counter() - return_start:.4f}s")

    print("=" * 40)
    print(f"TOTAL REQUEST TIME: {time.perf_counter() - total_start:.2f}s")
    print("=" * 40)

    return response