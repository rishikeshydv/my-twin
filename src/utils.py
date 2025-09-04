import requests
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
        # print("Embedding from Ollama:", data)
        return data["embedding"]
    
    except requests.exceptions.RequestException as e:
        print("Error communicating with Ollama:", e)