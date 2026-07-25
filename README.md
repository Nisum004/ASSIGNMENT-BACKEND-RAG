# Assignment Backend RAG

# About Me
Hello! I am Nisum Yonghang, currently studying 8th Semester CSIT at St.Xavier's College, Maitighar. I am an AI/ML enthusiast and this is my assignment for the role of AI/ML Intern at Palm Minds AI.

# About this Application:

A FastAPI backend that does two things:

1. Ingests documents — extracts text, chunks it, embeds it, and stores it in a vector database.
2. Chats about them — a conversational RAG endpoint that answers questions using the ingested documents, remembers conversation history, and can also handle a simple "book an interview" flow through natural language.

# STEPS:

1. Clone it

```
git clone https://github.com/Nisum004/ASSIGNMENT-BACKEND-RAG.git
```

2. Set up a virtual environment

```
python3 -m venv .venv
source .venv/bin/activate      
pip install --upgrade pip
pip install -e ".[dev]"
```

3. Configure your environment

Copy the example env file and fill in your own API keys.
```
cp .env.example .env
```

Redis instance running (used to remember chat history and in-progress bookings between messages). Locally:
```
docker run -d -p 6379:6379 redis
```
That matches the default `REDIS_URL=redis://localhost:6379/0` in `.env.example`.

4. Run it

```
uvicorn app.main:app --reload
```

The API is now live at `http://127.0.0.1:8000`. 
docs:
http://127.0.0.1:8000/docs

5. Try it out

A. Ingest a document
Upload a PDF or TXT file. It gets chunked, embedded, and stored in Pinecone.

```
curl -X POST 'http://127.0.0.1:8000/api/v1/documents/ingest?chunk_strategy=recursive' \
  -H 'accept: application/json' \
  -F 'file=@/path/to/your/document.pdf;type=application/pdf'
```

`chunk_strategy` can be `recursive` or `fixed` .

B. Chat about your documents

```
curl -X POST 'http://127.0.0.1:8000/api/v1/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "candidate-01",
    "message": "What experience does this candidate have with backend development?",
    "top_k": 4
  }'
```

C. Book an interview through chat

```
curl -X POST 'http://127.0.0.1:8000/api/v1/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "candiate-01",
    "message": "Book an Interview with Nisum Yonghang . his email is 022bscit026@sxc.edu.np for 2026-07-27 at 1:00",
    "top_k": 4
  }'
```

If you leave something out , it'll ask for what's missing instead of failing:
Just reply with the missing details in your next message (same `session_id`) and it'll pick up where it left off.

# Project layout:
```
app/
├── api/v1/          
├── services/         
├── repositories/      
├── models/            
├── schemas/            
└── core/config.py       
```
