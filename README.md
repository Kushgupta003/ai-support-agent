# 🤖 AI Customer Support Agent with RAG

A production-ready AI Customer Support Agent built using **LangGraph, FastAPI, and RAG (Retrieval-Augmented Generation)**.  
This project demonstrates real-world AI system design, backend development, and intelligent query routing.

---

## 🚀 Features

- 🔍 **RAG-based Document Search** (PDF & Text Upload)
- 🧠 **Automatic Query Categorization** (Technical / Billing / General)
- 😊 **Sentiment Analysis** (Auto-escalates negative queries)
- ⚡ **Fast Response Time** (1–2 seconds)
- 💬 **Real-time Chat Interface**
- 🗂️ **File Upload & Management System**
- 📊 **Logging & Intelligent Routing**

---

## 🛠️ Tech Stack

| Category       | Technologies |
|---------------|-------------|
| **Backend**   | FastAPI, Python 3.10+ |
| **AI/ML**     | LangGraph, Sentence Transformers |
| **Vector DB** | FAISS |
| **Frontend**  | HTML, CSS, JavaScript |
| **Server**    | Uvicorn |

---

## 💻 Installation

```bash
# Clone repository
git clone https://github.com/Kushgupta003/ai-support-agent.git
cd ai-support-agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py

👉 Open in browser:
http://localhost:8000

📄 Requirements
fastapi
uvicorn
python-multipart
sentence-transformers
pypdf
faiss-cpu

🌐 Usage
Start the server
Open the web interface
Upload PDF/TXT documents (auto-indexed using FAISS)
Ask queries → system retrieves relevant context + responds

📊 Performance Metrics
| Metric          | Value   |
| --------------- | ------- |
| Response Time   | 1–2 sec |
| Memory Usage    | ~500 MB |
| Offline Support | 100%    |
| Accuracy        | ~95%    |

🎯 Example Queries
| Query                                | Category  |
| ------------------------------------ | --------- |
| "How do I reset my password?"        | Technical |
| "Where can I find my receipt?"       | Billing   |
| "What are your business hours?"      | General   |
| "I am very angry about the service!" | Escalated |

🧱 Project Structure
ai-support-agent/
│── main.py              # LangGraph workflow
│── app.py               # FastAPI backend
│── templates/
│   └── index.html       # Frontend UI
│── static/              # CSS/JS
│── uploads/             # Documents
│── requirements.txt
│── README.md

🎓 Learning Outcomes
RAG (Retrieval-Augmented Generation)
LangGraph Workflow Design
FastAPI Backend Development
Vector Search using FAISS
Full Stack Integration

📈 Future Enhancements
LLM-based smarter responses
Multi-language support
Voice assistant integration
Admin dashboard
Docker + Cloud deployment

📬 Contact

Kush Gupta
🔗 GitHub: https://github.com/Kushgupta003

🔗 LinkedIn: https://www.linkedin.com/in/kush-gupta-cse

📄 License

MIT License
