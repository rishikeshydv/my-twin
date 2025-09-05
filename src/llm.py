import requests
       
def promptEngineering(inputText:str)->str:
    prompt = f"""
        User Input: {inputText}
    You are a Planner Agent.

    Analyze the user's input and return exactly one JSON object with the following six keys only:

    "category" → either "startup" or "code"

    "sub_category" → if "startup" then "sales & marketing" or "research"; if "code" then "optimize" or "add"

    "improved_query" → rewrite the input in clear, professional English

    "mcp_server" → "mentor" if related to guidance/strategy/startup; "code" if related to programming/technical tasks

    "vector_database" → "startup" if about startups/sales/marketting/research; "code" if about programming/technical tasks

    "web_search" → true if external info is required; otherwise false

    Do not include extra keys, categories, or nested objects. Return exactly one flat JSON object.
    """
    return prompt

def LLMResponse(inputText:str)->str:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model":"llama3.2",
        "messages":[{"role":"user","content":inputText}],
        "stream":False
    }
    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()

        data = res.json()
        content = data["message"]["content"]
        return content
    except requests.exceptions.RequestException as e:
        print("Error communicating with Ollama:", e)
 
def createEmbeddings(input: str) -> list[float]:
    try:
        res = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "all-minilm",
                "prompt": input
            }
        )
        res.raise_for_status()
        data = res.json()
        return data["embedding"]
    
    except requests.exceptions.RequestException as e:
        print("Error communicating with Ollama:", e)