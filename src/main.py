import os

from contextlib import asynccontextmanager
import threading
from src.llm import LLMResponse, promptEngineering, createEmbeddings
from src.file_handle import PDFHandler
from src.voice import record_and_detect, save_wav, transcribe
from src.mcp import WebSearchMCP, DeepSearchMCP
from src.models import getStartupContext
from fastapi import FastAPI
from pydantic import BaseModel
from watchdog.observers import Observer
import time
import json

def run_transcription():
    try:
        for utterance in record_and_detect():
            wav_path = save_wav(utterance)
            transcript = transcribe(wav_path)
            print("\n📝 Transcript:", transcript.text)
            print("Getting LLM response")
            llm_response = LLMResponse(promptEngineering(transcript.text))  #gets json of plan
            print("💬 LLM Response:", llm_response)
            query_map = json.loads(llm_response)
            if query_map["category"] == "startup":
                #get startup info context
                startupContext = getStartupContext()
                if query_map["web_search"] == "True":
                    print("🌐 Performing Deep + Web Search")
                    res = WebSearchMCP(query_map["improved_query"],startupContext)
                    print("Web Search MCP Result:", res)
                else:
                    print("Performing Deep Search Only")
                    res = DeepSearchMCP(query_map["improved_query"],startupContext)
                    print("Deep Search MCP Result:", res)
                    
                
    except KeyboardInterrupt:
        print("\n🛑 Trancription Stopped")
    
def start_pdf_watcher():
    folder_to_watch = "feed_me/new"
    os.makedirs(folder_to_watch, exist_ok=True)
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    observer.start()
    print(f"👀 Watching folder: {folder_to_watch} for new PDFs")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    t1 = threading.Thread(target=run_transcription, daemon=True)
    t1.start()
    
    t2 = threading.Thread(target=start_pdf_watcher, daemon=True)
    t2.start()
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