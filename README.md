# Production AI RAG Engine

A production-style Retrieval-Augmented Generation (RAG) system that demonstrates the complete lifecycle of an enterprise AI application—from document ingestion and hybrid retrieval to response generation, evaluation, and deployment.

Built using **Hybrid Retrieval (BM25 + Dense Search)**, **CrossEncoder Reranking**, **Amazon Bedrock**, **FastAPI**, and a modular architecture inspired by production AI systems.

> 🚧 **Project Status**
>
> This project is under active development. Additional features, documentation, architecture diagrams, and benchmarks will continue to be added.

---

## Highlights

- 🚀 Hybrid Retrieval (BM25 + Dense Search)
- 🎯 CrossEncoder Reranking
- 🤖 Amazon Bedrock (Nova Lite)
- 🌊 Streaming REST API
- ⚡ FastAPI
- 📄 PDF Document Ingestion Pipeline
- 📊 Automated Evaluation Framework inspired by **RAGAS**
- 🐳 Dockerized Deployment
- 📝 Structured Logging
- 🏗️ Modular Production Architecture

---

# Architecture

> Architecture diagram coming soon.

```text
                 Documents (PDF)
                        │
                        ▼
                  PDF Parsing
                        │
                        ▼
                   Chunking
                        │
                        ▼
             Sentence Embeddings
                        │
                        ▼
                  FAISS Index
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
   Dense Retrieval                BM25 Search
         │                             │
         └──────────────┬──────────────┘
                        ▼
                Hybrid Retrieval
                        │
                        ▼
            CrossEncoder Reranker
                        │
                        ▼
              Prompt Construction
                        │
                        ▼
         Amazon Bedrock (Nova Lite)
                        │
                        ▼
                 Generated Answer
```

---

# Capabilities

## Document Processing

- PDF document ingestion
- Intelligent document chunking
- Sentence Transformer embeddings
- Automatic vector index generation

---

## Retrieval

- Hybrid Retrieval (BM25 + Dense Search)
- FAISS Vector Search
- CrossEncoder Reranking
- Configurable Top-K Retrieval

---

## Generation

- Amazon Bedrock Nova Lite
- Context-aware prompt construction
- Streaming API responses
- Source-grounded answer generation

---

## Evaluation

The project includes an automated evaluation framework inspired by the **RAGAS methodology** for measuring both retrieval quality and answer quality.

Current metrics include:

- Recall@K
- Mean Reciprocal Rank (MRR)
- nDCG
- Faithfulness

---

## Deployment

- FastAPI REST API
- Docker Support
- Configuration Management
- Structured Logging

---

# Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.12 |
| API | FastAPI |
| LLM | Amazon Bedrock (Nova Lite) |
| Embeddings | Sentence Transformers |
| Dense Retrieval | FAISS |
| Sparse Retrieval | BM25 |
| Reranking | CrossEncoder |
| Evaluation | RAGAS-inspired Evaluation Framework |
| Containerization | Docker |

---

# Project Structure

```text
rag-engine/
│
├── data/
│   ├── documents/
│   ├── indexes/
│   └── benchmarks/
│
├── src/
│   ├── api/
│   ├── app/
│   ├── chunking/
│   ├── config/
│   ├── embedding/
│   ├── evaluation/
│   ├── generation/
│   ├── indexing/
│   ├── ingestion/
│   ├── parsing/
│   ├── prompting/
│   ├── reranking/
│   ├── retrieval/
│   ├── storage/
│   └── utils/
│
├── tools/
│
├── app.py
├── evaluate.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Getting Started

## Prerequisites

- Python 3.12+
- Docker Desktop (Recommended)
- AWS CLI configured
- Access to Amazon Bedrock

---

## Clone the Repository

```bash
git clone https://github.com/bikash-singhal/<repository-name>.git

cd <repository-name>
```

---

## Configure Environment Variables

Copy

```text
.env.example
```

to

```text
.env
```

Update the following values:

```env
AWS_PROFILE=<your_profile>

AWS_REGION=us-east-1

BEDROCK_GENERATION_MODEL=amazon.nova-lite-v1:0

BEDROCK_EVALUATION_MODEL=amazon.nova-lite-v1:0
```

---

# Docker (Recommended)

Build and start the application.

```bash
docker compose up --build
```

> **Note**
>
> Update the AWS credentials volume in `docker-compose.yml` to point to your local AWS credentials directory.
>
> **Windows**
>
> ```
> C:\Users\<username>\.aws
> ```
>
> **Linux/macOS**
>
> ```
> ${HOME}/.aws
> ```

---

# Local Development

Create a virtual environment.

### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Running the Application

## Interactive CLI

```bash
python app.py
```

The application automatically:

- Loads documents
- Builds the vector index (if required)
- Starts an interactive RAG session

---

## REST API

Start the FastAPI server.

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Interactive API documentation:

```
http://localhost:8000/docs
```

Health endpoint:

```
GET /health
```

> Swagger UI screenshots will be added soon.

---

# Evaluation Framework

Run the benchmark suite.

```bash
python evaluate.py
```

## Benchmark Dataset

Amazon SageMaker Documentation

---

## Results

| Metric | Score |
|---------|------:|
| Recall@K | **1.000** |
| Mean Reciprocal Rank (MRR) | **0.833** |
| nDCG | **0.907** |
| Faithfulness | **0.849** |

---

## Overall Performance

| Metric | Value |
|---------|------:|
| Test Cases | 10 |
| Passed | 10 |
| Failed | 0 |
| Success Rate | **100%** |

> Evaluation screenshots and benchmark reports will be added soon.

---

# Roadmap

Planned enhancements include:

- Authentication & Authorization
- CI/CD Pipeline
- Kubernetes Deployment
- Observability & Monitoring
- Query Expansion
- Hybrid Score Fusion Experiments
- Additional RAGAS-inspired Evaluation Metrics
- Multi-document Collections

---

# Author

**Bikash Singhal**

GitHub

https://github.com/bikash-singhal

LinkedIn

https://www.linkedin.com/in/bikashsinghal

---

# License

This project is licensed under the **MIT License**.