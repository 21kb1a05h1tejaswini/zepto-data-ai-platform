import os
import glob
from typing import List, TypedDict
from fastapi import FastAPI
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

# Initialize ChromaDB and Embeddings
client = chromadb.Client()
collection = client.get_or_create_collection(name="zepto_policies")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Ingest corpus
def ingest_docs():
    if collection.count() == 0:
        doc_files = sorted(glob.glob("docs/doc_*.txt"))
        for file_path in doc_files:
            doc_id = os.path.basename(file_path).replace(".txt", "")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            embedding = embedder.encode([text])[0].tolist()
            collection.add(
                ids=[doc_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"source": doc_id}]
            )

ingest_docs()

# Schemas
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float

class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_docs: List[str]
    retrieved_ids: List[str]
    response: QueryResponse

# LangGraph Nodes
KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

def classify_intent(state: AgentState):
    query = state["query"].lower()
    if any(kw in query for kw in KEYWORDS):
        intent = "policy_question"
    else:
        intent = "general_question"
    return {"intent": intent}

def route_intent(state: AgentState):
    return state["intent"]

def retrieve_and_answer(state: AgentState):
    query_emb = embedder.encode([state["query"]])[0].tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    top_doc = results["documents"][0][0]
    top_ids = results["ids"][0]
    
    snippet = top_doc[:200]
    ans_text = f"Based on the retrieved context: {snippet}"
    response = QueryResponse(answer=ans_text, sources=top_ids, confidence=1.0)
    return {"retrieved_docs": results["documents"][0], "retrieved_ids": top_ids, "response": response}

def direct_answer(state: AgentState):
    response = QueryResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )
    return {"response": response}

# Graph Compilation
workflow = StateGraph(AgentState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)
agent = workflow.compile()

# FastAPI App
app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=QueryResponse)
def ask_endpoint(req: QueryRequest):
    result = agent.invoke({"query": req.query})
    return result["response"]
