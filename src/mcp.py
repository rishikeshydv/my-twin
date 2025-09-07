import time
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from src.llm import LLMResponse
import ast
from dotenv import load_dotenv
import os

load_dotenv()
AZURE_CLIENT_ENDPOINT = os.getenv("AZURE_CLIENT_ENDPOINT")
AZURE_AGENT_ID = os.getenv("AZURE_AGENT_ID")

# web search for mcp
def WebSearchMCP(query: str, context:str)-> str:
    # Initialize the AI Project client
    project = AIProjectClient(
        endpoint=AZURE_CLIENT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )

    # Load the agent
    agent = project.agents.get_agent(AZURE_AGENT_ID)
    print(f"✅ Agent loaded: {agent.name}")

    # Create a new thread
    thread = project.agents.threads.create()
    print(f"🧵 Thread started: {thread.id}")

    # Prompt engineering for Bing-based search agent
    prompt = f"""
    You are a Web Research Agent that specializes in finding recent, credible, and relevant information to support a startup's strategy and decision-making. 
    Your focus is on queries related to the startup's industry, market trends, competitors, customer behavior, regulations, and emerging technologies.
    
    ### Context:
    The following context is relevant from our internal knowledge base:
    {context}
      
    Guidelines:
        1. Always perform a Bing Web Search before responding.
        2. Prioritize information from credible sources:
        - Established media outlets (e.g., NYT, TechCrunch, WSJ, Wired).
        - Industry reports (e.g., McKinsey, Gartner, Deloitte).
        - Government or regulatory websites (.gov, .eu, .org).
        - Company press releases, blogs, and reputable niche industry sources.
        3. Filter for **recent information** (last 12-18 months unless historical context is explicitly required).
        4. Present findings as a structured list:
        - Title of source
        - Author/organization
        - Publication date
        - URL
        - 2-3 sentence summary explaining why it is relevant to the startup.
        5. If multiple perspectives exist (e.g., conflicting market projections), include at least 2 contrasting credible sources.
        6. Do not fabricate links or data. Only use verifiable sources.
        7. When possible, highlight **insights or implications for the startup**, not just raw information.

    Query: {query}
        """.strip()

    # Send the user message
    message = project.agents.messages.create(thread.id, role="user", content=prompt)
    print(f"📤 Prompt sent, Message ID: {message.id}")

    # Start a run
    run = project.agents.runs.create(thread_id=thread.id, agent_id=agent.id)
    print(f"▶️ Run started: {run.id}")

    # Poll until run is complete
    while run.status in ("queued", "in_progress"):
        time.sleep(1)
        run = project.agents.runs.get(thread.id, run.id)

    if run.status == "failed":
        print("❌ Run failed:", run.last_error)
        return

    print(f"✅ Run completed with status: {run.status}")

    # Collect messages
    structured_messages = ""
    messages = project.agents.messages.list(thread.id, order="asc")
    for m in messages:
        content = next(
            (c for c in m.content if c.type == "text" and hasattr(c, "text")), None
        )
        if content and m.role == "assistant":
            structured_messages += content.text.value + "\n\n"
    return structured_messages

# llm call to synthesize sub-query responses
def SynthesizeOverview(main_query: str, subquery_responses: list[str]) -> str:
    joined_responses = "\n\n---\n\n".join(subquery_responses)

    prompt = f"""
    You are a research assistant tasked with synthesizing multiple detailed research responses into one cohesive report.

    ### Main Query:
    {main_query}

    ### Inputs (sub-query responses):
    {joined_responses}

    ### Instructions:
    1. Write a **single, structured overview** that integrates the insights from all three responses.
    2. Avoid repetition — merge overlapping points into a unified perspective.
    3. Organize the report with clear **sections and headings**.
    4. Highlight:
       - Key findings
       - Major trends or insights
       - Connections between the sub-queries
       - Gaps or open questions
    5. End with a **concise conclusion** (3-4 sentences) summarizing the overall answer.

    ### Output Format:
    - Title
    - Integrated overview (structured with sections + bullet points)
    - Final conclusion
    """
    resp = LLMResponse(prompt)
    return resp

#deep search for mcp
def DeepSearchMCP(query: str, context:str):
    prompt = f"""
            You are a research assistant that breaks down complex queries into smaller, targeted sub-queries for web search.
            
            ### Context:
            The following context is relevant from our internal knowledge base:
            {context}

            ### Instructions:
            - Output must be a **valid JSON list of exactly 3 strings**.
            - Each sub-query must focus on a different aspect of the main query.
            - Sub-queries should be clear, specific, and non-overlapping.
            - Do not include explanations, labels, or extra text — only return the JSON.

            ### Example
            Input: best strategies for startup growth in 2024
            Output: ["startup growth strategies 2024", "marketing tactics for startups 2024", "funding options for startups 2024"]

            ### Task
            Input: {query}
            Output:
            """

    response = LLMResponse(prompt)
    sub_queries = ast.literal_eval(response)
    print("🔍 Sub-queries generated:", sub_queries)

    # storing sub-queries' responses
    subquery_responses = []
    # performing llm search for each sub-query
    for sq in sub_queries:
        search_prompt = f"""
            You are an AI research assistant tasked with performing a deep exploration of the following research query:
            Query: "{sq}"
            
                        
            ### Context:
            The following context is relevant from our internal knowledge base:
            {context}


            ### Instructions:
            1. Provide a **detailed, structured answer** to the query.
            2. Cover multiple perspectives, dimensions, or factors relevant to the query.
            3. If useful, include:
            - Key trends
            - Examples or case studies
            - Data points (if known)
            - Challenges or limitations
            - Future outlook
            4. Organize your response with clear **headings and bullet points** for readability.
            5. Do not include irrelevant information or generic filler.

            ### Output Format:
            - Title (short summary of the query focus)
            - Structured analysis (with sections + bullet points)
            - Brief conclusion (2-3 sentences with key takeaways)
            """
        print(f"📌 Researching: {sq}")
        detailed_response = LLMResponse(search_prompt)
        subquery_responses.append((sq, detailed_response))

    # synthesizing sub-query responses
    overview = SynthesizeOverview(query, [resp for _, resp in subquery_responses])
    print("📝 Synthesized Overview:\n", overview)
    return overview
