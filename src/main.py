# from dotenv import load_dotenv
# import os

# load_dotenv()
# API_KEY = os.getenv("API_KEY")
from contextlib import asynccontextmanager
import threading
from src.llm import LLMResponse, promptEngineering
from src.utils import createEmbeddings
from src.file_handle import extractPDF
from src.voice import record_and_detect, save_wav, transcribe
from fastapi import FastAPI
from pydantic import BaseModel

def run_transcription():
    try:
        for utterance in record_and_detect():
            wav_path = save_wav(utterance)
            transcript = transcribe(wav_path)
            print("\n📝 Transcript:", transcript.text)
            print("Getting LLM response")
            llm_response = LLMResponse(promptEngineering(transcript.text))
            print("💬 LLM Response:", llm_response) 
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    t = threading.Thread(target=run_transcription, daemon=True)
    t.start()
    yield
    # Shutdown (if needed, cleanup here)
    print("👋 FastAPI shutting down")

app = FastAPI(lifespan=lifespan)

#structure from request Data
class RequestData(BaseModel):
    input: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/t2t")
def text_to_text(data:RequestData):
    print("Received text:", data.input)
    response = LLMResponse(data.input)
    return {"response": response}

#test embedding
@app.post("/t2e")
def text_to_embedding(data:RequestData):
    print("Received text for embedding:", data.input)
    embedding = createEmbeddings(data.input)
    return {"embedding": embedding}