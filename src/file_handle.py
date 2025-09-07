from PyPDF2 import PdfReader
from src.models import createTable, storeInfo
from src.llm import LLMResponse, createEmbeddings
import ast
from watchdog.events import FileSystemEventHandler

def extractPDF(file_path:str, tableName:str, contentType)->None:
    reader = PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        print(text)
        prompt = f"""
        You are an Information Extractor Agent.
        The user provides text (from a PDF page).
        Your task is to read the text carefully and return only the important points or information as a list of strings
        and nothing else. Don't return all the points but only the most crucial and important points. Strictly only return the list of important notes without any textual description before or after.

        Rules:

        Focus only on meaningful insights, facts, or takeaways.

        Ignore filler words, repetitions, disfluencies (like "um," "uh," "you know"), and irrelevant details.

        Summarize long sentences into concise, clear bullet points.

        Output format must be a JSON list of strings.
        
        PDF Text:
        {text}
        """
        createTable(tableName)
        importantNotes = LLMResponse(prompt)
        importantNotesList = ast.literal_eval(importantNotes)
        for note in importantNotesList:
            embedding = createEmbeddings(note)
            storeInfo(tableName, contentType, note, embedding)   
            
    print(f"PDF '{file_path}' processed and data stored in table '{tableName}'.")
            
class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".pdf"):
            print(f"New PDF detected: {event.src_path}")
            #get file name
            file_name = event.src_path.split("/")[-1].split(".")[0]
            extractPDF(event.src_path, "startupinfo",file_name.lower())
        