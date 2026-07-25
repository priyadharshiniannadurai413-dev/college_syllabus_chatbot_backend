import os
import asyncio
import litellm
from litellm import completion
import edge_tts
from app.core.config import settings

async def speech_to_text(audio_path: str) -> str:
    """
    Converts speech from an audio file to text using Groq's whisper-large-v3-turbo model.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # Use litellm.atranscription for async transcription
    with open(audio_path, 'rb') as audio_file:
        output = await litellm.atranscription(
            model="groq/whisper-large-v3-turbo", 
            file=audio_file,
            # api_key=settings.groq_api_key
        )
    
    # response = completion(
    #     model='gemini/gemini-2.5-flash',
    #     messages= [
    #         {"role": "system", "content": "summarize the text short and crisp"},
    #         {"role": "user", "content": output.text}
    #     ]
    # )
    # return response.choices[0].message.content
    return output.text

async def text_to_speech(text: str, output_path: str, voice: str = 'en-IN-PrabhatNeural') -> None:
    """
    Converts text to speech using edge_tts and saves it to output_path.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

async def main():
    input_audio = 'recordings/question.mp3.mp4'
    output_audio = 'outputs/output.mp3'
    
    print(f"1. Performing STT on '{input_audio}'...")
    try:
        text = await speech_to_text(input_audio)
        print(f"STT Output (Transcribed Text): \"{text}\"\n")
    except Exception as e:
        print(f"Error during STT: {e}")
        return

    print(f"2. Performing TTS to generate '{output_audio}'...")
    try:
        await text_to_speech(text, output_audio)
        print(f"TTS Output successfully saved to: {output_audio}")
    except Exception as e:
        print(f"Error during TTS: {e}")

if __name__ == "__main__":
    asyncio.run(main())