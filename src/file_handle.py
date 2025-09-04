from PyPDF2 import PdfReader
from src.models import createTable, storeInfo
from src.llm import LLMResponse

def extractPDF(file_path:str, tableName:str):
    reader = PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        print(text)
        # prompt = f"""
        # You are an Information Extractor Agent.
        # The user provides text (from a PDF page).
        # Your task is to read the text carefully and return only the important points or key information as a list.

        # Rules:

        # Focus only on meaningful insights, facts, or takeaways.

        # Ignore filler words, repetitions, disfluencies (like "um," "uh," "you know"), and irrelevant details.

        # Summarize long sentences into concise, clear bullet points.

        # Output format must be a JSON list of strings.
        
        # PDF Text:
        # {text}
        # """
        #createTable(tableName)
        #importantNotes = LLMResponse(prompt)
        #print("Extracted Important Notes:", importantNotes)
        #storeInfo(tableName, "startup", prompt, [0.0]*768)
        
        