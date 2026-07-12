# RAG Engine

A production-style Retrieval-Augmented Generation (RAG) application built from scratch using:

- Python
- Sentence Transformers
- FAISS
- OpenAI
- Docker

---

# Features

- PDF document ingestion
- Intelligent text chunking
- Sentence Transformer embeddings
- FAISS vector search
- Prompt engineering
- OpenAI-powered answer generation
- Dockerized deployment

---

# Quick Start (Docker) ⭐ Recommended

Docker is the recommended way to run this application.

## Prerequisites

- Docker Desktop

---

## 1. Clone the Repository

```bash
git clone <repository-url>

cd rag-engine
```

---

## 2. Configure Environment Variables

Copy

```text
.env.example
```

to

```text
.env
```

Example:

```text
OPENAI_API_KEY=your_openai_api_key_here

# Optional
HF_TOKEN=your_huggingface_token
```

---

## 3. Build the Docker Image

```bash
docker build -t rag-engine:v1 .
```

---

## 4. Run the Application

### Windows (PowerShell)

```powershell
docker run -it `
--env-file .env `
rag-engine:v1 `
--pdf data/raw/next-generation-sagemaker-ug.pdf `
--question "What is Amazon Bedrock?"
```

### Linux / macOS

```bash
docker run -it \
--env-file .env \
rag-engine:v1 \
--pdf data/raw/next-generation-sagemaker-ug.pdf \
--question "What is Amazon Bedrock?"
```

---

# Local Development (Optional)

This section is intended for contributors and developers.

## Create a Virtual Environment

### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

### Windows (PowerShell)

```powershell
python app.py `
--pdf data/raw/next-generation-sagemaker-ug.pdf `
--question "What is Amazon Bedrock?"
```

### Linux / macOS

```bash
python app.py \
--pdf data/raw/next-generation-sagemaker-ug.pdf \
--question "What is Amazon Bedrock?"
```

---

# Running Tests

```bash
python -m tests.sanity_check
```

---

# Project Structure

```text
rag-engine/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── faiss_store.py
│   ├── llm.py
│   ├── models.py
│   ├── pdf_reader.py
│   ├── prompt_builder.py
│   └── rag_pipeline.py
│
├── tests/
│   ├── __init__.py
│   └── sanity_check.py
│
├── .dockerignore
├── .env.example
├── Dockerfile
├── app.py
├── config.py
├── README.md
└── requirements.txt
```

---

# Technologies Used

- Python
- Sentence Transformers
- FAISS
- OpenAI API
- Docker

---

# Roadmap

Planned improvements include:

- AWS Bedrock integration
- Anthropic Claude support
- Hugging Face models
- FastAPI REST API
- Streamlit UI
- Persistent vector database
- Separate ingestion and querying pipelines
- Incremental document indexing
- Conversation memory
- Hybrid search
- CI/CD pipeline
- Unit testing

---

# Learning Objectives

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- Embeddings
- Vector Databases
- Prompt Engineering
- Large Language Models
- Docker
- Production-ready Python project structure

---

# License

This project is intended for educational purposes.