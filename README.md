# Agentic PDF Search System - API-First Compliance Assistant

This is an end-to-end, production-ready AI Assistant designed for ingestions and hybrid semantic searches of enterprise compliance and regulatory documents (e.g. SEBI/NSE circulars). It features a multi-agent execution workflow orchestrated via **LangGraph**, backed by **Qdrant** (vector store) and **Elasticsearch** (full-text metadata store), utilizing local vision models (`qwen3-vl:latest`) for complex graph/table reading.

This project is optimized to run natively on **Apple Silicon (M1/M2/M3/M4/M5 macOS)**. The frontend has been removed, and the entire workspace is controllable via the built-in interactive **Swagger UI**.

---

## 🛠️ System Architecture

* **Backend**: FastAPI (Python 3.12)
* **Agent Graph**: LangGraph orchestrating 8 agents (Query Understanding, Planning, Hybrid Search, Validator, Vision, Synthesis, Citations, Critic)
* **Vector Engine**: Qdrant (cos-similarity, `bge-m3` vectors)
* **Full-text/Metadata search**: Elasticsearch (BM25 keyword search)
* **Relational Storage**: PostgreSQL (document cataloging, page classifications, logs, agent traces)
* **Caching & Rates**: Redis
* **AI Models**: Local Ollama execution of `qwen3` (reasoning) and `qwen3-vl` (vision)

---

## 💻 Terminal Setup Guide (From Scratch on M5 Macbook)

Follow these steps to set up and run the project from scratch on your new macOS machine.

### Step 1: Xcode Command Line Tools
Open your terminal and make sure Xcode command line tools are installed:
```bash
xcode-select --install
```

### Step 2: Install Homebrew (Package Manager)
If you do not have Homebrew installed, paste this in your terminal:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Add Homebrew to your PATH by following the instructions displayed in your terminal output.

### Step 3: Install Docker Desktop (Recommended)
Docker is the simplest way to run databases. Install it via Homebrew:
```bash
brew install --cask docker
```
Open **Docker Desktop** from your Applications folder to start the Docker daemon.

### Step 4: Run Infrastructure (Docker Compose)
Start PostgreSQL, Redis, Qdrant, and Elasticsearch in the background:
```bash
docker compose -f docker/docker-compose.yml up -d
```
Verify the containers are running:
```bash
docker ps
```

### Step 5: Install & Run Ollama (Local AI Engines)
1. Install Ollama:
   ```bash
   brew install ollama
   ```
2. Start the Ollama background service:
   ```bash
   brew services start ollama
   ```
3. Pull the required models (BGE-M3 for embeddings, Qwen3 for reasoning, and Qwen3-VL for tables/graphs):
   ```bash
   # Pull Reasoning LLM
   ollama pull qwen3:latest
   
   # Pull Vision LLM
   ollama pull qwen3-vl:latest
   
   # Pull Embeddings model
   ollama pull bge-m3
   ```

### Step 6: Create Python Virtual Environment
Initialize python 3.12 and install backend dependencies:
```bash
# Verify Python version (should be 3.12+)
python3 --version

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r backend/requirements.txt
```

### Step 7: Launch FastAPI Server
Ensure you are in the workspace root, set the Python Path environment variable, and start the development server:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 backend/app/main.py
```
You will see output indicating uvicorn is running on:
👉 **`http://localhost:8000`**

---

## 🚀 Interacting with Swagger UI

FastAPI automatically parses endpoint descriptions and inputs.

1. Open your browser and navigate to: **[http://localhost:8000/docs](http://localhost:8000/docs)** (or simply [http://localhost:8000/](http://localhost:8000/)).
2. You will be greeted by the **Swagger UI** containing all available REST endpoints.
3. Click on any route, select **"Try it out"**, fill in the parameters, and click **"Execute"** to run API requests live!

---

## 📡 Terminal Command Line Testing (Curl Examples)

If you prefer testing from the command line, open a new terminal tab and run these templates:

### 1. Check System Health
Validate connections to Postgres, Redis, Qdrant, and Elasticsearch:
```bash
curl -X GET "http://localhost:8000/api/health"
```

### 2. Ingest a Compliance PDF
Upload a PDF (e.g. SEBI circular) and execute classification pipelines:
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@/path/to/your/document.pdf" \
  -F "name=Margin Collection Circular" \
  -F "circular_number=SEBI/HO/109/2026" \
  -F "issue_date=2026-06-17" \
  -F "department=SEBI" \
  -F "tags=margin,compliance"
```

### 3. Ask Agentic Search Query
Submit a query. This will kick off the LangGraph multi-agent hybrid retrieval loop:
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the margin collection guidelines for f&o trades?",
    "session_id": "auditor-session-100"
  }'
```

### 4. Fetch Agent Execution Trace Logs
Fetch the exact logs and state changes of all 8 agents generated during the search session:
```bash
curl -X GET "http://localhost:8000/api/agent-trace?session_id=auditor-session-100"
```

### 5. List all circulars in Database
```bash
curl -X GET "http://localhost:8000/api/documents"
```

### 6. Delete a Circular
Permanently wipe metadata and clean vector/full-text indexes across PostgreSQL, Qdrant, and Elasticsearch:
```bash
curl -X DELETE "http://localhost:8000/api/document/<document-uuid>"
```
