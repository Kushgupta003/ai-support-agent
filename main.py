import os
import re
from typing import Dict, TypedDict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str
    context: str

# Global Embeddings
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # Pre-cached embeddings (downloaded once)
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings

class DocumentProcessor:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = None
        
    def load_pdf(self, file_path: str) -> None:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        self._process_documents(documents)
        
    def load_text(self, file_path: str) -> None:
        loader = TextLoader(file_path)
        documents = loader.load()
        self._process_documents(documents)
        
    def _process_documents(self, documents: List) -> None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        else:
            self.vectorstore.add_documents(texts)
            
    def search(self, query: str, k: int = 3) -> str:
        if self.vectorstore is None:
            return "No documents loaded yet."
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context if context else "No relevant context found."

# ✅ Rule-Based Sentiment Analysis (No Model Download)
def analyze_sentiment_local(query: str) -> str:
    negative_words = ['angry', 'hate', 'terrible', 'worst', 'bad', 'fail', 'broken', 'not working', 'error', 'issue', 'problem']
    positive_words = ['great', 'good', 'excellent', 'perfect', 'love', 'helpful', 'thank', 'thanks', 'awesome']
    
    query_lower = query.lower()
    
    neg_count = sum(1 for word in negative_words if word in query_lower)
    pos_count = sum(1 for word in positive_words if word in query_lower)
    
    if neg_count > pos_count:
        return "Negative"
    elif pos_count > neg_count:
        return "Positive"
    else:
        return "Neutral"

# ✅ Rule-Based Categorization (No Model Download)
def categorize_query(query: str) -> str:
    query_lower = query.lower()
    
    # Technical keywords
    technical_keywords = ['internet', 'connection', 'wifi', 'password', 'login', 'technical', 'error', 'bug', 'not working', 'broken', 'crash', 'freeze', 'slow', 'loading', 'app', 'software', 'device', 'hardware', 'setup', 'install', 'update']
    
    # Billing keywords
    billing_keywords = ['billing', 'payment', 'invoice', 'receipt', 'refund', 'charge', 'price', 'cost', 'bill', 'subscription', 'cancel', 'upgrade', 'downgrade', 'plan', 'credit', 'debit', 'card', 'money', 'pay', 'purchased', 'purchase']
    
    # Count matches
    tech_count = sum(1 for word in technical_keywords if word in query_lower)
    bill_count = sum(1 for word in billing_keywords if word in query_lower)
    
    if tech_count > bill_count:
        return "Technical"
    elif bill_count > tech_count:
        return "Billing"
    else:
        return "General"

# ✅ Pre-defined Response Templates (No Model Download)
def generate_response(category: str, query: str, context: str) -> str:
    query_lower = query.lower()
    
    # Technical Responses
    if category == "Technical":
        if "password" in query_lower or "login" in query_lower:
            return "To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and follow the instructions sent to your inbox."
        elif "internet" in query_lower or "connection" in query_lower or "wifi" in query_lower:
            return "For internet connection issues, please try restarting your router. If the problem persists, check your network settings or contact your ISP."
        elif "app" in query_lower or "software" in query_lower:
            return "For app issues, please try updating to the latest version. If the problem continues, try uninstalling and reinstalling the app."
        elif "error" in query_lower or "broken" in query_lower or "not working" in query_lower:
            return "I apologize for the technical issue. Please provide more details about the error message you're seeing so I can assist you better."
        else:
            return "I'd be happy to help with your technical query. Could you please provide more specific details about the issue you're experiencing?"
    
    # Billing Responses
    elif category == "Billing":
        if "receipt" in query_lower or "invoice" in query_lower:
            return "You can find your receipt in your account under 'Order History' or check your email for the confirmation email from your purchase."
        elif "refund" in query_lower:
            return "For refund requests, please contact our billing team with your order number. Refunds are typically processed within 5-7 business days."
        elif "payment" in query_lower or "bill" in query_lower or "charge" in query_lower:
            return "You can view your billing information in your account dashboard. For payment issues, please contact our billing support team."
        elif "cancel" in query_lower:
            return "To cancel your subscription, go to your account settings and click 'Cancel Subscription'. You'll receive a confirmation email shortly."
        else:
            return "I'd be happy to help with your billing query. Could you please provide more specific details about your billing concern?"
    
    # General Responses
    else:
        if "hours" in query_lower or "time" in query_lower:
            return "Our business hours are Monday to Friday, 9 AM to 6 PM. We're closed on weekends."
        elif "contact" in query_lower or "phone" in query_lower or "email" in query_lower:
            return "You can reach us at support@example.com or call us at 1-800-123-4567 during business hours."
        elif "location" in query_lower or "address" in query_lower:
            return "Our office is located at 123 Main Street, City, State 12345. You can also find us on Google Maps."
        else:
            return "Thank you for reaching out! I'm here to help. Could you please provide more details about your query?"

# Node Functions
def categorize(state: State) -> State:
    category = categorize_query(state["query"])
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    sentiment = analyze_sentiment_local(state["query"])
    return {"sentiment": sentiment}

def get_context(state: State) -> State:
    processor = DocumentProcessor()
    context = processor.search(state["query"])
    return {"context": context}

def handle_technical(state: State) -> State:
    response = generate_response("Technical", state["query"], state.get("context", ""))
    return {"response": response}

def handle_billing(state: State) -> State:
    response = generate_response("Billing", state["query"], state.get("context", ""))
    return {"response": response}

def handle_general(state: State) -> State:
    response = generate_response("General", state["query"], state.get("context", ""))
    return {"response": response}

def escalate(state: State) -> State:
    return {"response": "This query has been escalated to a human agent due to its negative sentiment."}

def route_query(state: State) -> str:
    if state["sentiment"] == "Negative":
        return "escalate"
    elif state["category"] == "Technical":
        return "handle_technical"
    elif state["category"] == "Billing":
        return "handle_billing"
    else:
        return "handle_general"

# Graph Construction
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("get_context", get_context)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "get_context")
workflow.add_conditional_edges("get_context", route_query, {
    "handle_technical": "handle_technical",
    "handle_billing": "handle_billing",
    "handle_general": "handle_general",
    "escalate": "escalate"
})
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)
workflow.set_entry_point("categorize")

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

doc_processor = DocumentProcessor()

def run_customer_support(query: str, thread_id: str = "1") -> Dict[str, str]:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"query": query, "category": "", "sentiment": "", "response": "", "context": ""}
    results = app.invoke(initial_state, config=config)
    return {
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": results["response"],
        "context": results["context"]
    }