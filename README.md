# 🤖 RAG-Based AI Customer Support Agent

A full-stack AI Customer Support Agent built with **LangGraph, FastAPI, FAISS, Hugging Face Sentence Transformers, and Ollama**.

The system uses **Retrieval-Augmented Generation (RAG)** to answer customer questions from uploaded documents while reducing hallucinations through context-grounded responses.

---

## 🚀 Features

* 🔍 **RAG-based Document Search**

  * Upload PDF knowledge-base documents
  * Extract and split document content into chunks
  * Generate semantic embeddings
  * Store and retrieve relevant chunks using FAISS

* 🧠 **Intelligent Query Categorization**

  * Technical
  * Billing
  * General

* 😊 **Sentiment Analysis**

  * Positive
  * Neutral
  * Negative
  * Negative queries can trigger human-support escalation

* 🕸️ **LangGraph Workflow**

  * Query categorization
  * Sentiment analysis
  * Context retrieval
  * Conditional routing
  * Grounded response generation
  * Fallback handling
  * Escalation handling

* 🤖 **Local LLM with Ollama**

  * Qwen2.5 3B
  * Runs locally without requiring an external LLM API

* 🛡️ **Grounded Responses**

  * Answers are generated using retrieved document context
  * If relevant information cannot be found, the system returns a fallback response instead of intentionally inventing an answer

* 💬 **Real-Time Web Chat Interface**

* 📂 **Document Management**

  * Upload documents
  * List uploaded documents
  * Delete documents
  * Rebuild the vector store

* 💾 **Persistent FAISS Index**

  * Vector index is saved locally
  * Existing indexes can be restored when the application starts

* 🌐 **REST API**

  * FastAPI-based backend
  * Chat and document-management endpoints

---

## 🏗️ System Architecture

```text
                   ┌─────────────────────┐
                   │    User / Web UI    │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │      FastAPI        │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │     LangGraph       │
                   │   Workflow Engine   │
                   └──────────┬──────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
        Categorization   Sentiment        Retrieval
              │               │                │
              │               │                ▼
              │               │        ┌──────────────┐
              │               │        │     FAISS    │
              │               │        └──────┬───────┘
              │               │               │
              │               │               ▼
              │               │        Relevant Context
              │               │               │
              └───────────────┴───────────────┤
                                              ▼
                                   ┌────────────────────┐
                                   │    Qwen2.5 3B      │
                                   │       Ollama       │
                                   └──────────┬─────────┘
                                              │
                                              ▼
                                      Grounded Response
```

---

## 🧠 RAG Pipeline

```text
PDF Document
     ↓
PyPDFLoader
     ↓
Text Extraction
     ↓
Recursive Character Text Splitter
     ↓
Sentence Transformer Embeddings
     ↓
FAISS Vector Store
     ↓
Semantic Similarity Search
     ↓
Relevant Context
     ↓
Qwen2.5 3B
     ↓
Final Support Response
```

The system uses:

**Embedding Model**

```text
sentence-transformers/all-MiniLM-L6-v2
```

**LLM**

```text
qwen2.5:3b
```

---

## 🕸️ LangGraph Workflow

The support workflow is implemented as a state graph:

```text
User Query
    ↓
Categorize
    ↓
Analyze Sentiment
    ↓
Retrieve Context
    ↓
 ┌───────────────────────┐
 │ Relevant information? │
 └───────────┬───────────┘
             │
       ┌─────┴─────┐
       │           │
      Yes          No
       │           │
       ▼           ▼
Generate       Fallback
Answer         Response
       │
       ▼
Escalation Check
       │
       ▼
Final Response
```

---

## 🛠️ Tech Stack

| Category        | Technologies                       |
| --------------- | ---------------------------------- |
| Language        | Python                             |
| Backend         | FastAPI                            |
| ASGI Server     | Uvicorn                            |
| AI Workflow     | LangGraph                          |
| LLM             | Ollama + Qwen2.5 3B                |
| Embeddings      | Hugging Face Sentence Transformers |
| Vector Database | FAISS                              |
| PDF Processing  | PyPDF                              |
| Text Splitting  | LangChain Text Splitters           |
| Frontend        | HTML, CSS, JavaScript              |
| API Validation  | Pydantic                           |

---

## 💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Kushgupta003/ai-support-agent.git
cd ai-support-agent
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama on your system and make sure the Ollama server is running.

Then download the required model:

```bash
ollama pull qwen2.5:3b
```

Verify the model:

```bash
ollama list
```

You should see:

```text
qwen2.5:3b
```

### 5. Start the Application

```bash
python app.py
```

Or:

```bash
python -m uvicorn app:app --reload
```

Open:

```text
http://localhost:8000
```

---

## 📄 Usage

### Step 1 — Start the application

Run the FastAPI server.

### Step 2 — Upload a knowledge-base document

Upload a PDF containing company or customer-support information.

The system automatically:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS Index
```

### Step 3 — Ask questions

Example:

```text
What are the payment methods?
```

The system retrieves the relevant section from the uploaded document and generates a grounded response.

### Example

**User:**

```text
What are the payment methods?
```

**AI:**

```text
We accept:
- Credit Cards
- Debit Cards
- UPI
- Net Banking
- PayPal
```

---

## 📡 API Endpoints

| Method | Endpoint                | Description              |
| ------ | ----------------------- | ------------------------ |
| GET    | `/`                     | Web interface            |
| POST   | `/upload`               | Upload a document        |
| POST   | `/chat`                 | Ask the AI support agent |
| GET    | `/documents`            | List uploaded documents  |
| DELETE | `/documents/{filename}` | Delete a document        |

### Chat Request

```json
{
  "query": "What are the payment methods?",
  "thread_id": "1"
}
```

### Example Response

```json
{
  "category": "Billing",
  "sentiment": "Neutral",
  "response": "We accept Credit Cards, Debit Cards, UPI, Net Banking, and PayPal.",
  "context": "..."
}
```

---

## 🧪 Example Queries

| Query                                | Expected Category |
| ------------------------------------ | ----------------- |
| "How do I reset my password?"        | Technical         |
| "What are the payment methods?"      | Billing           |
| "How can I download my invoice?"     | Billing           |
| "How do I upgrade my plan?"          | Billing           |
| "What are your business hours?"      | General           |
| "I am very angry about the service!" | Escalation        |

---

## 📁 Project Structure

```text
ai-support-agent/
│
├── main.py                    # LangGraph workflow and RAG pipeline
├── app.py                     # FastAPI backend
├── requirements.txt           # Python dependencies
├── README.md
├── procfile
│
├── templates/
│   └── index.html             # Web interface
│
├── static/                    # Static frontend assets
│
├── uploads/                   # Uploaded documents
│
└── data/
    └── faiss/                 # Persisted FAISS vector index
```

---

## 🔐 Environment Configuration

The application supports environment variables for configuration.

| Variable           | Default                                  | Description                |
| ------------------ | ---------------------------------------- | -------------------------- |
| `EMBEDDING_MODEL`  | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model            |
| `OLLAMA_MODEL`     | `qwen2.5:3b`                             | Local LLM                  |
| `OLLAMA_BASE_URL`  | `http://localhost:11434`                 | Ollama server              |
| `RAG_MAX_DISTANCE` | `1.2`                                    | FAISS relevance threshold  |
| `RAG_TOP_K`        | `3`                                      | Number of retrieved chunks |
| `FAISS_DIR`        | `data/faiss`                             | FAISS index location       |

---

## 🎯 Key Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* Vector similarity search
* FAISS vector databases
* Sentence Transformer embeddings
* LangGraph state-based workflows
* Local LLM integration with Ollama
* FastAPI REST APIs
* PDF document processing
* Prompt-based grounded generation
* Sentiment-based escalation
* Conditional AI routing
* Persistent vector storage
* Full-stack AI application integration

---

## 🔮 Future Enhancements

* 🔹 LLM-based query classification
* 🔹 Advanced ML sentiment analysis
* 🔹 Multi-language support
* 🔹 Conversation memory improvements
* 🔹 Streaming LLM responses
* 🔹 Authentication and role-based access
* 🔹 Admin dashboard
* 🔹 Docker containerization
* 🔹 Cloud deployment
* 🔹 Automated evaluation and RAG benchmarking
* 🔹 Support for additional document formats

---

## 👨‍💻 Author

**Kush Gupta**

GitHub:
https://github.com/Kushgupta003

LinkedIn:
https://www.linkedin.com/in/kush-gupta-cse

---

## 📄 License

This project is licensed under the MIT License.
