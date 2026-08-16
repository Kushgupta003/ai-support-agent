import os
from typing import Dict, TypedDict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# FAISS uses L2 distance by default. Lower distance = more similar.
# Tune this value after testing your own documents.
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "1.2"))

TOP_K = int(os.getenv("RAG_TOP_K", "3"))

FAISS_DIR = os.getenv("FAISS_DIR", "data/faiss")


# ============================================================
# LangGraph State
# ============================================================

class State(TypedDict):
    query: str
    category: str
    sentiment: str
    context: str
    response: str
    sources: list
    should_escalate: bool
    retrieval_relevant: bool


# ============================================================
# Embeddings
# ============================================================

_embeddings = None


def get_embeddings():
    """Create the embedding model once and reuse it."""
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embeddings


# ============================================================
# Document / RAG Processing
# ============================================================

class DocumentProcessor:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self.last_sources = []
        self.last_retrieval_relevant = False

        # Try to restore an existing FAISS index.
        self.load_index()

    def load_pdf(self, file_path: str) -> None:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        self._process_documents(documents)

    def load_text(self, file_path: str) -> None:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        self._process_documents(documents)

    def _process_documents(self, documents: List) -> None:
        """Split documents and add them to the FAISS vector store."""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        texts = text_splitter.split_documents(documents)

        if not texts:
            return

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(
                texts,
                self.embeddings,
            )
        else:
            self.vectorstore.add_documents(texts)

        self.save_index()

    def search(
        self,
        query: str,
        k: int = TOP_K,
    ) -> Tuple[str, list, bool]:
        """
        Retrieve the most relevant document chunks.

        Returns:
            context: combined relevant text
            sources: document/page metadata
            relevant: whether retrieval passed the distance threshold
        """

        self.last_sources = []
        self.last_retrieval_relevant = False

        if self.vectorstore is None:
            return (
                "No documents loaded yet.",
                [],
                False,
            )

        # similarity_search_with_score returns lower L2 distance
        # for more similar documents.
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=max(1, k),
        )

        if not results:
            return (
                "No relevant context found.",
                [],
                False,
            )

        relevant_results = [
            (doc, score)
            for doc, score in results
            if score <= RAG_MAX_DISTANCE
        ]

        if not relevant_results:
            return (
                "No relevant context found.",
                [],
                False,
            )

        context_parts = []
        sources = []

        for doc, score in relevant_results:
            context_parts.append(doc.page_content.strip())

            metadata = doc.metadata or {}

            source = {
                "file": os.path.basename(
                    metadata.get("source", "Unknown document")
                ),
                "page": (
                    metadata.get("page", 0) + 1
                    if isinstance(metadata.get("page"), int)
                    else metadata.get("page")
                ),
                "distance": round(float(score), 4),
            }

            sources.append(source)

        context = "\n\n---\n\n".join(
            part for part in context_parts if part
        )

        self.last_sources = sources
        self.last_retrieval_relevant = bool(context)

        print("\n========== QUERY ==========")
        print(query)

        print("\n========== RETRIEVED CHUNKS ==========")
        print(context)

        print("\n========== SOURCES ==========")
        print(sources)

        return (
            context if context else "No relevant context found.",
            sources,
            self.last_retrieval_relevant,
        )

    def save_index(self) -> None:
        """Persist FAISS locally so it survives application restarts."""

        if self.vectorstore is None:
            return

        os.makedirs(FAISS_DIR, exist_ok=True)
        self.vectorstore.save_local(FAISS_DIR)

    def load_index(self) -> None:
        """Load a previously saved FAISS index if it exists."""

        index_file = os.path.join(FAISS_DIR, "index.faiss")
        metadata_file = os.path.join(FAISS_DIR, "index.pkl")

        if not (
            os.path.exists(index_file)
            and os.path.exists(metadata_file)
        ):
            return

        try:
            self.vectorstore = FAISS.load_local(
                FAISS_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            print("Loaded persisted FAISS index.")

        except Exception as exc:
            print(f"Could not load FAISS index: {exc}")
            self.vectorstore = None


# ============================================================
# Local Sentiment Analysis
# ============================================================

def analyze_sentiment_local(query: str) -> str:
    """
    Lightweight rule-based sentiment detection.

    This is intentionally simple and should be described as
    rule-based sentiment analysis, not ML sentiment analysis.
    """

    negative_words = [
        "angry",
        "hate",
        "terrible",
        "worst",
        "bad",
        "fail",
        "failed",
        "broken",
        "not working",
        "error",
        "issue",
        "problem",
        "frustrated",
        "frustrating",
        "unacceptable",
        "disappointed",
    ]

    positive_words = [
        "great",
        "good",
        "excellent",
        "perfect",
        "love",
        "helpful",
        "thank",
        "thanks",
        "awesome",
    ]

    query_lower = query.lower()

    neg_count = sum(
        1 for word in negative_words
        if word in query_lower
    )

    pos_count = sum(
        1 for word in positive_words
        if word in query_lower
    )

    if neg_count > pos_count:
        return "Negative"

    if pos_count > neg_count:
        return "Positive"

    return "Neutral"


# ============================================================
# Query Categorization
# ============================================================

def categorize_query(query: str) -> str:
    """Lightweight rule-based support category detection."""

    query_lower = query.lower()

    technical_keywords = [
        "internet",
        "connection",
        "wifi",
        "password",
        "login",
        "technical",
        "error",
        "bug",
        "not working",
        "broken",
        "crash",
        "freeze",
        "slow",
        "loading",
        "app",
        "software",
        "device",
        "hardware",
        "setup",
        "install",
        "update",
    ]

    billing_keywords = [
        "billing",
        "payment",
        "invoice",
        "receipt",
        "refund",
        "charge",
        "price",
        "cost",
        "bill",
        "subscription",
        "cancel",
        "upgrade",
        "downgrade",
        "plan",
        "credit",
        "debit",
        "card",
        "money",
        "pay",
        "purchased",
        "purchase",
    ]

    tech_count = sum(
        1 for word in technical_keywords
        if word in query_lower
    )

    bill_count = sum(
        1 for word in billing_keywords
        if word in query_lower
    )

    if tech_count > bill_count:
        return "Technical"

    if bill_count > tech_count:
        return "Billing"

    return "General"


# ============================================================
# LLM
# ============================================================

_llm = None


def get_llm():
    """
    Lazily create the local Ollama LLM.

    Make sure Ollama is running and the selected model exists:
        ollama pull tinyllama
    """

    global _llm

    if _llm is None:
        _llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )

    return _llm


def generate_response(
    query: str,
    category: str,
    sentiment: str,
    context: str,
) -> str:
    """
    Generate a grounded answer using only retrieved context.

    If there is no relevant context, do not invent an answer.
    """

    if (
        not context
        or context == "No documents loaded yet."
        or context == "No relevant context found."
    ):
        return (
            "I couldn't find relevant information in the uploaded "
            "documents to answer that question accurately."
        )

    prompt = f"""
You are a customer support assistant.

Answer the user's question using ONLY the information in the context.

IMPORTANT:
- Extract the exact information needed to answer the question.
- Do not omit important steps, names, methods, or details from the context.
- If the answer contains steps, include all relevant steps in the correct order.
- Do not invent or add information.
- Do not explain these instructions.
- Keep the answer concise.

If the context does not contain enough information to answer the question, say:
"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{query}

Answer:
""".strip()

    try:
        result = get_llm().invoke(prompt)

        if hasattr(result, "content"):
            response = result.content
        else:
            response = str(result)

        return response.strip()

    except Exception as exc:
        print(f"LLM generation error: {exc}")

        return (
            "The AI response service is currently unavailable. "
            "Please make sure the local Ollama server is running "
            f"and that the '{LLM_MODEL}' model is installed."
        )


# ============================================================
# LangGraph Nodes
# ============================================================

def categorize(state: State) -> dict:
    return {
        "category": categorize_query(state["query"])
    }


def analyze_sentiment(state: State) -> dict:
    return {
        "sentiment": analyze_sentiment_local(state["query"])
    }


def get_context(state: State) -> dict:
    global doc_processor

    context, sources, relevant = doc_processor.search(
        state["query"],
        k=TOP_K,
    )

    return {
        "context": context,
        "sources": sources,
        "retrieval_relevant": relevant,
    }


def generate_answer(state: State) -> dict:
    response = generate_response(
        query=state["query"],
        category=state["category"],
        sentiment=state["sentiment"],
        context=state["context"],
    )

    return {
        "response": response
    }


def escalation_check(state: State) -> dict:
    """
    Generate the answer first, then add escalation for negative
    sentiment instead of replacing the useful answer.
    """

    should_escalate = state["sentiment"] == "Negative"

    response = state["response"]

    if should_escalate:
        response += (
            "\n\nI understand that this situation is frustrating. "
            "I've flagged this conversation for human support."
        )

    return {
        "response": response,
        "should_escalate": should_escalate,
    }


def route_after_retrieval(state: State) -> str:
    """
    If retrieval has no relevant information, go directly to
    the fallback node. Otherwise generate a grounded answer.
    """

    if not state.get("retrieval_relevant", False):
        return "fallback"

    return "generate_answer"


def fallback_response(state: State) -> dict:
    return {
        "response": (
            "I couldn't find relevant information in the uploaded "
            "documents to answer that question accurately. "
            "Please upload a relevant support document or contact "
            "a human support agent."
        ),
        "should_escalate": state["sentiment"] == "Negative",
    }


# ============================================================
# Graph Construction
# ============================================================

workflow = StateGraph(State)

workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("get_context", get_context)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("fallback", fallback_response)
workflow.add_node("escalation_check", escalation_check)

workflow.set_entry_point("categorize")

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "get_context")

workflow.add_conditional_edges(
    "get_context",
    route_after_retrieval,
    {
        "generate_answer": "generate_answer",
        "fallback": "fallback",
    },
)

workflow.add_edge("generate_answer", "escalation_check")

workflow.add_edge("escalation_check", END)
workflow.add_edge("fallback", END)


# MemorySaver keeps LangGraph checkpoints for thread IDs.
checkpointer = MemorySaver()

app = workflow.compile(
    checkpointer=checkpointer
)


# ============================================================
# Global Document Processor
# ============================================================

doc_processor = DocumentProcessor()


# ============================================================
# Public API Used by app.py
# ============================================================

def run_customer_support(
    query: str,
    thread_id: str = "1",
) -> Dict:
    """Run one customer-support query through the LangGraph workflow."""

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    initial_state: State = {
        "query": query,
        "category": "",
        "sentiment": "",
        "context": "",
        "response": "",
        "sources": [],
        "should_escalate": False,
        "retrieval_relevant": False,
    }

    results = app.invoke(
        initial_state,
        config=config,
    )

    return {
        "category": results.get("category", "General"),
        "sentiment": results.get("sentiment", "Neutral"),
        "response": results.get("response", ""),
        "context": results.get("context", ""),
        "sources": results.get("sources", []),
        "should_escalate": results.get(
            "should_escalate",
            False,
        ),
    }
