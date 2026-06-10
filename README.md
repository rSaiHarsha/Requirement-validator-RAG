# Requirement Validation RAG System

An advanced, premium Streamlit dashboard for automotive requirements verification. This application utilizes Retrieval-Augmented Generation (RAG) to evaluate systems engineering specifications against compliance reference manuals (e.g., the INCOSE Guide to Writing Requirements) using high-performance embedding models and LLMs.

---

## 🛠️ Architecture & Technologies

- **Frontend Interface**: Streamlit (with customized responsive layouts and clean horizontal navigation).
- **LLM Engine**: NVIDIA NIM OpenAI API (`meta/llama-3.1-70b-instruct`).
- **Vector Database**: Qdrant Cloud / Local client.
- **Embedding Model**: NVIDIA NIM Embedding API (`nvidia/nv-embedqa-e5-v5`).
- **PDF Extraction**: PyMuPDF (`fitz`).

---

## 📁 File Structure

```text
Requirement-Validation-Rag/
├── app.py               # Main Streamlit dashboard code (ingestor & validator tabs)
├── requirements.txt     # Locked project dependencies
├── .env                 # Environment variables (API keys and DB URLs)
└── .gitignore           # Standard git ignoring rules (excludes .env & .venv)
```

---

## ⚙️ Prerequisites

1. **Python**: Python 3.10 to 3.12 (standard virtual env ready).
2. **NVIDIA API Key**: Obtain from the [NVIDIA API Catalog](https://build.nvidia.com/) to access LLM and Embedding models.
3. **Qdrant Vector Database**: A running instance (Qdrant Cloud URL and API key, or a local dockerized deployment).

---

## 🚀 Setup & Installation

Follow these steps to set up and run the project locally on your system:

### 1. Clone or Move to Project Folder
Make sure you are in the project folder containing the code:
```bash
cd Requirement-Validation-Rag
```

### 2. Create a Virtual Environment
Initialize a fresh Python virtual environment to keep packages isolated:
```bash
python -m venv .venv
```

Activate the virtual environment:
- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
Install all required package dependencies defined in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` in the root of the project folder (it should be in the same folder as `app.py`). Use this structure:

```env
NVIDIA_API_KEY=your-nvidia-nim-api-key-here
QDRANT_URL=your-qdrant-cluster-url-here
QDRANT_API_KEY=your-qdrant-api-key-here
```

---

## 🖥️ Running the Application

Launch the Streamlit dashboard server with:
```bash
streamlit run app.py
```

Once started, the application will automatically open in your browser, typically at:
[http://localhost:8501/](http://localhost:8501/)

---

## 📖 Usage Guide

### Tab 1: 📁 Upload Knowledge (Document Ingestion)
1. **Upload Reference**: Upload standard reference materials (like regulatory specifications or compliance manuals) in **PDF** format.
2. **Configure Settings**: A dialog box will automatically open asking you to select whether you want to **Add to Existing Collection** or **Create New Collection**, and to set the **Page Range** extraction parameters.
3. **Confirm & Extract**: Click **Confirm & Extract Chunks**. The application will read the text, use `meta/llama-3.1-70b-instruct` to extract clean standalone semantic paragraphs, and list them in a collapsible **Chunks Created** expander.
4. **Ingest to Vector DB**: Click **Upload Chunks to vectorDB** to embed and save them in Qdrant.

### Tab 2: ☑️ Search & Retrieval (Requirements Validator)
1. **Enter Query**: Type an automotive/ADAS requirement you wish to critique in the input field, or click one of the **Quick Suggestions** buttons below the search bar.
2. **Choose Source Collection**: Click **Analyze**. A dialog will open asking you to select the vector database source you want to query guidelines from (e.g., `All Collections`, a specific database collection, or `None` for a zero-shot LLM check).
3. **Review Compliance Critique**: Read the generated compliance report critiquing the input requirement based on INCOSE standards, including suggestions on how to rewrite it correctly.
4. **View Retrieved Guidelines**: Expand the **View Retrieved Guidelines** card below the compliance critique to inspect the actual database chunks retrieved from Qdrant, including similarity scores, type badges, page numbers, and keywords.
