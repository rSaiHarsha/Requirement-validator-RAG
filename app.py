import streamlit as st
import fitz                          # pymupdf
from openai import OpenAI            # NVIDIA NIM
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)
import os
import re
import uuid
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded (with override to fetch the latest .env changes)
load_dotenv(override=True)

NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "").strip()
QDRANT_URL      = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY  = os.environ.get("QDRANT_API_KEY", "").strip()

EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
EMBED_DIM       = 1024
BATCH_SIZE      = 8
DEFAULT_COLLECTION = "incose_gtwr_v4"
CHAT_MODEL      = "meta/llama-3.1-70b-instruct"

# Page configurations - Collapse sidebar by default (and CSS will completely hide it)
st.set_page_config(
    page_title="INCOSE GtWR Suite",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom premium styling (Glassmorphism, custom horizontal tabs, and sidebar hiding)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* Completely hide the sidebar navigation and toggle controls */
[data-testid="stSidebar"] {
    display: none;
}
[data-testid="stSidebarCollapsedControl"] {
    display: none;
}

/* Page titles */
.main-header {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.5rem;
    margin-bottom: 0.2rem;
    text-align: center;
}

.sub-header {
    font-size: 1.1rem;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 2rem;
}

/* Custom Chunk Cards */
.chunk-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(8px);
}

.chunk-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 10px 20px rgba(99, 102, 241, 0.1);
}

.chunk-header {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
}

.chunk-badge {
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
}

/* Validator Statuses */
.chunk-badge.score {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #ffffff;
}

.chunk-badge.section {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

.chunk-badge.item {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.chunk-badge.page {
    background: rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

/* Ingestor Statuses */
.chunk-badge.pending {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.25);
}

.chunk-badge.ingested {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
}

.chunk-badge.error {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.25);
}

.chunk-badge.type {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.25);
}

.chunk-body {
    font-size: 0.925rem;
    color: #cbd5e1;
    line-height: 1.6;
}

/* Tag style */
.tag-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}

.keyword-tag {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    color: #94a3b8;
}

/* Style normal buttons */
div.stButton > button {
    transition: all 0.2s ease;
}

/* Custom premium horizontal tabs styling */
div[data-baseweb="tab-list"] {
    background-color: #0f172a !important; /* Dark Slate 900 */
    border-bottom: 2px solid #1e293b !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    gap: 24px !important;
    margin-bottom: 25px !important;
}

div[data-baseweb="tab-list"] button[role="tab"] {
    color: #94a3b8 !important; /* Muted Slate */
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    background: none !important;
    border: none !important;
    padding: 10px 16px !important;
    border-radius: 0px !important;
    margin-bottom: -2px !important; /* Overlap container border to prevent double lines */
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease-in-out !important;
}

div[data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #ffffff !important; /* White */
    border-bottom: 2px solid #ef4444 !important; /* Red highlight underline - forms a single line with the container border */
    font-weight: 700 !important;
}

div[data-baseweb="tab-list"] button[role="tab"]:hover {
    color: #cbd5e1 !important;
}

</style>
""", unsafe_allow_html=True)

# Initialize API clients
@st.cache_resource
def init_clients():
    if not NVIDIA_API_KEY or not QDRANT_URL:
        return None, None
    nvidia = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return nvidia, qdrant

nvidia, qdrant = init_clients()

# Shared helpers
def embed_passages(texts: list[str]) -> list[list[float]]:
    if not nvidia:
        raise ValueError("NVIDIA API client is not initialized.")
    response = nvidia.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "END"},
    )
    return [item.embedding for item in response.data]

def embed_query(text: str) -> list[float]:
    if not nvidia:
        raise ValueError("NVIDIA API client is not initialized.")
    response = nvidia.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"},
    )
    return response.data[0].embedding

def extract_json_array(llm_output: str) -> list[dict]:
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', llm_output, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = llm_output.strip()
        
    start_idx = json_str.find('[')
    end_idx = json_str.rfind(']')
    if start_idx != -1 and end_idx != -1:
        json_str = json_str[start_idx:end_idx+1]
        
    try:
        data = json.loads(json_str, strict=False)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except Exception as e:
        st.error(f"Failed to parse JSON array from LLM response: {e}")
    return []

def generate_chunks_with_llm(client: OpenAI, page_text: str, page_num: int) -> list[dict]:
    if not page_text.strip():
        return []

    system_prompt = (
        "You are an expert systems engineer and technical writer. Your task is to read a raw text page "
        "extracted from a technical manual/specification and break it down into one or more high-quality, "
        "standalone text chunks optimized for semantic search and RAG.\n\n"
        "Instructions:\n"
        "1. Identify the logical topics, requirements, rules, configurations, or sections on the page.\n"
        "2. For each identified topic, write a clean, detailed text description containing all technical terms, "
        "codes, and variables. Retain full context so the chunk is self-explanatory.\n"
        "3. If there are tables, lists, or structured data, format them into clear Markdown tables/lists within the text.\n"
        "4. Assign appropriate metadata to each chunk.\n"
        "5. Output the result ONLY as a valid JSON array of objects. Do not include any commentary outside the JSON.\n\n"
        "Each object in the JSON array must follow this exact schema:\n"
        "[\n"
        "  {\n"
        "    \"title\": \"A descriptive title of the topic\",\n"
        "    \"text\": \"The detailed content containing technical details, codes, Markdown tables, etc. Retain full context.\",\n"
        "    \"metadata\": {\n"
        "      \"item_type\": \"requirement\", \"configuration\", \"architecture\", \"api\", \"definition\", or \"other\",\n"
        "      \"item_id\": \"Any specific ID found (e.g., SWS_Can_00123, R12, C4) or null\",\n"
        "      \"item_name\": \"The title or name of the requirement/topic\",\n"
        "      \"keywords\": [\"kw1\", \"kw2\", \"kw3\"]\n"
        "    }\n"
        "  }\n"
        "]"
    )

    user_prompt = f"Page Number: {page_num}\n\nRaw Page Text:\n{page_text}"

    retries = 3
    delay = 2
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            llm_output = response.choices[0].message.content
            chunks = extract_json_array(llm_output)
            if chunks:
                for c in chunks:
                    if not isinstance(c, dict):
                        continue
                    if "metadata" not in c or not isinstance(c["metadata"], dict):
                        c["metadata"] = {}
                    c["metadata"]["page"] = page_num
                    
                    if "text" not in c:
                        for key in ["description", "content", "body"]:
                            if key in c:
                                c["text"] = c[key]
                                break
                    if "text" not in c:
                        c["text"] = json.dumps(c)
                return chunks
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                sleep_time = delay * (2 ** attempt)
                time.sleep(sleep_time)
            else:
                break
                
    return [{
        "title": f"Page {page_num} Content",
        "text": page_text,
        "metadata": {
            "item_type": "page_text",
            "item_id": None,
            "item_name": f"Raw content of page {page_num}",
            "keywords": ["fallback"],
            "page": page_num
        }
    }]

def setup_qdrant_collection(client: QdrantClient, collection_name: str, recreate: bool = False):
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            client.delete_collection(collection_name)
        else:
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBED_DIM,
            distance=Distance.COSINE,
            on_disk=False,
        ),
    )

    for field in ["item_type", "item_id", "page"]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=f"metadata.{field}",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="text",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=2,
            max_token_len=40,
            lowercase=True,
        ),
    )

# Dialog decorator for target collection settings and extraction parameters
@st.dialog("Target Collection & Extraction Parameters")
def configure_target_collection_dialog(file_obj):
    st.write(f"📂 **File Uploaded:** `{file_obj.name}`")
    
    # 1. Target Collection Section
    st.markdown("### 🗃️ 1. Target Collection")
    collection_mode = st.radio(
        "Choose target option:",
        ["Add to Existing Collection", "Create New Collection"],
        key="dlg_collection_mode"
    )
    
    target_collection = ""
    if collection_mode == "Add to Existing Collection":
        collections = []
        if qdrant:
            try:
                collections = [c.name for c in qdrant.get_collections().collections]
            except Exception as e:
                st.error(f"Error fetching collections: {e}")
        
        if collections:
            target_collection = st.selectbox("Select Target Collection:", collections, key="dlg_select_col")
        else:
            st.warning("No existing collections found. Please select 'Create New Collection'.")
    else:
        raw_name = st.text_input("Enter New Collection Name:", placeholder="e.g. autosar_manuals", key="dlg_new_col_name").strip()
        target_collection = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name) if raw_name else ""
        if raw_name and target_collection != raw_name:
            st.info(f"Cleaned collection name to: `{target_collection}`")
            
    # 2. Extraction Parameters Section
    st.markdown("### ⚙️ 2. Extraction Parameters")
    try:
        pdf_bytes = file_obj.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        file_obj.seek(0)
    except Exception as e:
        total_pages = 0
        st.error(f"Error reading PDF: {e}")
        
    if total_pages > 0:
        st.info(f"Loaded specification contains {total_pages} pages.")
        col_start, col_end = st.columns(2)
        with col_start:
            start_page = st.number_input("Start Page:", min_value=1, max_value=total_pages, value=1, key="dlg_start_page")
        with col_end:
            end_page = st.number_input("End Page:", min_value=1, max_value=total_pages, value=min(3, total_pages), key="dlg_end_page")
            
        if start_page > end_page:
            st.error("Start Page cannot be greater than End Page.")
            extract_ready = False
        else:
            extract_ready = True
    else:
        extract_ready = False
            
    if st.button("🧱 Confirm & Extract Chunks", type="primary", disabled=not extract_ready, key="dlg_confirm_btn"):
        if not target_collection:
            st.error("Please enter/select a valid collection name.")
        else:
            st.session_state.target_collection_name = target_collection
            st.session_state.collection_mode = collection_mode
            st.session_state.start_page = int(start_page)
            st.session_state.end_page = int(end_page)
            st.session_state.dialog_completed = True
            st.session_state.run_extraction = True
            st.rerun()

# Dialog decorator for choosing collection/knowledge base in Tab 2
@st.dialog("Select Knowledge Base")
def choose_collection_dialog(query_text):
    st.write("🔍 **Requirement under evaluation:**")
    st.info(f"\"{query_text}\"")
    st.write("Please select which knowledge base source you want to retrieve compliance rules from:")
    
    # Build the collection list
    db_collections = ["All Collections", "None", DEFAULT_COLLECTION]
    if qdrant:
        try:
            raw_cols = [c.name for c in qdrant.get_collections().collections]
            for c in raw_cols:
                if c not in db_collections:
                    db_collections.append(c)
        except Exception:
            pass
            
    selected_kb = st.selectbox(
        "Select Knowledge Base:",
        db_collections,
        index=0,
        key="dlg_val_selectbox"
    )
    
    col_confirm, col_cancel = st.columns([1, 1])
    with col_confirm:
        if st.button("Confirm & Analyze", type="primary", key="dlg_val_confirm", use_container_width=True):
            st.session_state.query_collection = selected_kb
            st.session_state.run_analysis = True
            st.session_state.show_val_dialog = False
            st.rerun()
    with col_cancel:
        if st.button("Cancel", key="dlg_val_cancel", use_container_width=True):
            st.session_state.show_val_dialog = False
            st.rerun()

# ----------------- SESSION STATE SETUP -----------------
# Validator
if "query_val" not in st.session_state:
    st.session_state.query_val = ""
if "main_search_input" not in st.session_state:
    st.session_state.main_search_input = ""
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "show_val_dialog" not in st.session_state:
    st.session_state.show_val_dialog = False
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = None
if "llm_analysis" not in st.session_state:
    st.session_state.llm_analysis = None
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# Ingestor Progressive State
if "extracted_chunks" not in st.session_state:
    st.session_state.extracted_chunks = None
if "target_collection_name" not in st.session_state:
    st.session_state.target_collection_name = ""
if "dialog_completed" not in st.session_state:
    st.session_state.dialog_completed = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "run_extraction" not in st.session_state:
    st.session_state.run_extraction = False
if "start_page" not in st.session_state:
    st.session_state.start_page = 1
if "end_page" not in st.session_state:
    st.session_state.end_page = 1

# Chat History
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant", 
            "content": "Hello! I am your Nemotron Core AI companion. You can ask me anything about requirements engineering, INCOSE guidelines, or safety compliance."
        }
    ]

# ----------------- TOP-LEVEL TABS NAVIGATION -----------------
tab1, tab2 = st.tabs([
    "📁 Upload Knowledge", 
    "☑️ Search & Retrieval", 
])

# ==============================================================================
# TAB 1: 📁 RAG Knowledge Engine (Document Ingestion)
# ==============================================================================
with tab1:

    st.markdown('<div class="sub-header">Upload baseline reference docs, standard operations manuals, or past specifications.</div>', unsafe_allow_html=True)

    # File uploader shown first
    uploaded_file = st.file_uploader("Drop foundational files here", type=["pdf"])

    # Progressive disclosure logic
    if uploaded_file is None:
        # Reset state if uploader is cleared
        st.session_state.dialog_completed = False
        st.session_state.target_collection_name = ""
        st.session_state.extracted_chunks = None
        st.session_state.current_file = None
        st.session_state.run_extraction = False
    else:
        # If new file is uploaded, reset states to show dialog
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.dialog_completed = False
            st.session_state.extracted_chunks = None
            st.session_state.target_collection_name = ""
            st.session_state.run_extraction = False
            st.rerun()

        # Trigger collection configuration dialog if not yet completed
        if not st.session_state.dialog_completed:
            configure_target_collection_dialog(uploaded_file)
            
            st.warning("⚠️ Action Required: Configure collection settings and parameters inside the dialog.")
            if st.button("Open Settings Dialog", key="reopen_dlg_btn"):
                configure_target_collection_dialog(uploaded_file)

        # Main page handles running the extraction if triggered by the dialog
        if st.session_state.get("run_extraction", False):
            st.markdown("---")
            st.markdown("### 🧱 Processing Document...")
            st.write(f"📍 **Target Collection:** `{st.session_state.target_collection_name}`")
            st.write(f"📄 **Page Range:** {st.session_state.start_page} to {st.session_state.end_page}")
            
            # Read PDF and run chunking
            try:
                pdf_bytes = uploaded_file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                total_pages = len(doc)
                uploaded_file.seek(0)
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                total_pages = 0
                st.session_state.run_extraction = False
                st.stop()
                
            if total_pages > 0:
                target_collection = st.session_state.target_collection_name
                collection_mode = st.session_state.get("collection_mode", "Create New Collection")
                
                if collection_mode == "Create New Collection":
                    try:
                        setup_qdrant_collection(qdrant, target_collection, recreate=False)
                        st.success(f"Ensured collection `{target_collection}` is created with indexes.")
                    except Exception as e:
                        st.error(f"Failed to create collection: {e}")
                        st.session_state.run_extraction = False
                        st.stop()
                
                st.session_state.extracted_chunks = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                extracted_list = []
                pages_to_process = list(range(st.session_state.start_page, st.session_state.end_page + 1))
                
                for idx, page_num in enumerate(pages_to_process):
                    status_text.text(f"Extracting & chunking Page {page_num} of {st.session_state.end_page}...")
                    
                    try:
                        page_text = doc[page_num - 1].get_text("text")
                        page_chunks = generate_chunks_with_llm(nvidia, page_text, page_num)
                        
                        for chunk in page_chunks:
                            chunk["id"] = str(uuid.uuid4())
                            chunk["status"] = "pending"
                            extracted_list.append(chunk)
                    except Exception as e:
                        st.error(f"Error extracting page {page_num}: {e}")
                        
                    progress_bar.progress((idx + 1) / len(pages_to_process))
                    time.sleep(0.5)
                    
                status_text.text("Extraction completed!")
                st.session_state.extracted_chunks = extracted_list
                st.session_state.run_extraction = False
                st.rerun()

    # Extracted Chunks Preview displayed collapsible after chunks are loaded
    if st.session_state.extracted_chunks is not None:
        st.markdown("---")
        st.markdown("### 📝 Extracted Chunks Preview")
        st.write(f"Target Collection: `{st.session_state.target_collection_name}`")
        
        chunks = st.session_state.extracted_chunks
        pending_count = sum(1 for c in chunks if c["status"] == "pending")
        ingested_count = sum(1 for c in chunks if c["status"] == "ingested")
        
        st.write(f"**Total Chunks:** {len(chunks)} | **Pending:** {pending_count} | **Ingested:** {ingested_count}")
        
        col_bulk, col_clear = st.columns([1, 1])
        with col_bulk:
            if pending_count > 0:
                if st.button("📥 Upload Chunks to vectorDB", type="primary", use_container_width=True, key="ingest_all_btn"):
                    with st.spinner("Embedding and uploading chunks in batches..."):
                         try:
                             pending_list = [c for c in chunks if c["status"] == "pending"]
                             for i in range(0, len(pending_list), BATCH_SIZE):
                                 batch = pending_list[i : i + BATCH_SIZE]
                                 texts = [c["text"] for c in batch]
                                 embeddings = embed_passages(texts)
                                 
                                 points = [
                                     PointStruct(
                                         id=chunk["id"],
                                         vector=embedding,
                                         payload={
                                             "title": chunk.get("title", "Untitled"),
                                             "text": chunk["text"],
                                             "metadata": chunk["metadata"],
                                         },
                                     )
                                     for chunk, embedding in zip(batch, embeddings)
                                 ]
                                 qdrant.upsert(collection_name=st.session_state.target_collection_name, points=points)
                                 for chunk in batch:
                                     chunk["status"] = "ingested"
                             st.success("Successfully ingested all pending chunks!")
                             st.rerun()
                         except Exception as e:
                             st.error(f"Bulk ingestion failed: {e}")
            else:
                st.success("All chunks have been ingested successfully!")
                
        with col_clear:
            if st.button("🧹 Clear Extracted Chunks", use_container_width=True, key="clear_extracted_btn"):
                st.session_state.extracted_chunks = None
                st.session_state.target_collection_name = ""
                st.session_state.dialog_completed = False
                st.session_state.run_extraction = False
                st.rerun()
                
        st.markdown("---")
        
        # Single expander wrapping the entire list of chunks, clicking displays them open and close
        with st.expander("Chunks Created", expanded=False):
            for idx, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                title = chunk.get("title", "Untitled")
                text = chunk.get("text", "")
                item_type = meta.get("item_type", "N/A")
                item_id = meta.get("item_id") or "N/A"
                page = meta.get("page", "N/A")
                keywords = meta.get("keywords", [])
                status = chunk.get("status", "pending")
                
                status_label = "Pending"
                if status == "ingested":
                    status_label = "Ingested"
                elif status == "error":
                    status_label = "Error"
                    
                safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
                safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
                safe_item_id = item_id.replace("<", "&lt;").replace(">", "&gt;")
                
                tags_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords])
                
                card_html = f"""
                <div class="chunk-card">
                    <div class="chunk-header">
                        <span class="chunk-badge {status}">{status_label}</span>
                        <span class="chunk-badge page">Page {page}</span>
                        <span class="chunk-badge type">{item_type}</span>
                        <span class="chunk-badge page">ID: {safe_item_id}</span>
                    </div>
                    <div class="chunk-body">
                        <strong style="font-size:1.05rem; color:#f1f5f9;">{safe_title}</strong>
                        <p style="margin-top:8px; margin-bottom:8px;">{safe_text}</p>
                    </div>
                    <div class="tag-container">
                        {tags_html}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                col_btn, col_info = st.columns([1, 4])
                with col_btn:
                    if status == "pending":
                        if st.button(f"📥 Ingest Chunk {idx+1}", key=f"ingest_indiv_{chunk['id']}", use_container_width=True):
                            try:
                                embeddings = embed_passages([text])
                                point = PointStruct(
                                    id=chunk["id"],
                                    vector=embeddings[0],
                                    payload={
                                        "title": title,
                                        "text": text,
                                        "metadata": meta,
                                    },
                                )
                                qdrant.upsert(collection_name=st.session_state.target_collection_name, points=[point])
                                chunk["status"] = "ingested"
                                st.success(f"Ingested Chunk {idx+1}!")
                                st.rerun()
                            except Exception as e:
                                chunk["status"] = "error"
                                st.error(f"Failed to ingest: {e}")
                    elif status == "ingested":
                        st.markdown("<span style='color:#34d399; font-weight:600; padding:6px 0; display:inline-block;'>✓ Added to collection</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#f87171; font-weight:600; padding:6px 0; display:inline-block;'>✗ Ingestion error</span>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

    # Verification Tool at bottom of Tab 1
    if qdrant and st.session_state.target_collection_name:
        st.markdown("---")
        with st.expander("🔍 Verify Collection Ingests (Run Query on Target DB)"):
            test_query = st.text_input("Enter verification search query:", placeholder="e.g. Can driver payload size", key="test_query_in")
            limit = st.slider("Number of results to fetch:", min_value=1, max_value=5, value=2, key="test_limit_sld")
            
            if st.button("Search Vector DB", key="test_search_btn") and test_query:
                with st.spinner("Searching..."):
                    try:
                        q_vec = embed_query(test_query)
                        search_results = qdrant.query_points(
                            collection_name=st.session_state.target_collection_name,
                            query=q_vec,
                            limit=limit,
                            with_payload=True
                        )
                        
                        if not search_results.points:
                            st.info("No matching chunks found in Qdrant collection.")
                        else:
                            for rank, r in enumerate(search_results.points, 1):
                                payload = r.payload
                                meta = payload.get("metadata", {})
                                t = payload.get("title", "Untitled")
                                txt = payload.get("text", "")
                                p = meta.get("page", "N/A")
                                itype = meta.get("item_type", "N/A")
                                iid = meta.get("item_id", "N/A")
                                
                                st.markdown(f"""
                                **Match #{rank} (Similarity Score: {r.score:.4f})**
                                * **Title:** `{t}` | **Page:** `{p}` | **Type:** `{itype}` | **ID:** `{iid}`
                                * **Text:** {txt}
                                ---
                                """)
                    except Exception as e:
                        st.error(f"Search verification failed: {e}")

# ==============================================================================
# TAB 2: ☑️ Requirements Quality Analyst (Requirements Validator)
# ==============================================================================
with tab2:
    if st.session_state.get("pending_query") is not None:
        st.session_state.main_search_input = st.session_state.pending_query
        st.session_state.query_val = st.session_state.pending_query
        st.session_state.pending_query = None

    st.markdown('<div class="main-header">Search from your Knowledge base</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automotive Requirements Verification via Knowledge Retrieval</div>', unsafe_allow_html=True)

    # Test questions focusing on automotive/ADAS requirements
    TEST_QUESTIONS = [
        {
            "title": "Test Case 1: Lane-Keep Assist (Vague & Combined)",
            "question": (
                "Evaluate this ADAS requirement: 'The ADAS lane-keep assist system shall always keep the car in the lane "
                "and function safely at high speed.' What specific INCOSE guidelines does it violate, and how should it "
                "be rewritten?"
            )
        },
        {
            "title": "Test Case 2: Autonomous Emergency Braking (Verifiability & Ambiguity)",
            "question": (
                "How does the INCOSE guide help in formulating verifiable requirements for Autonomous Emergency Braking (AEB)? "
                "Evaluate this requirement: 'The vehicle must apply brakes immediately when an obstacle is close.'"
            )
        },
        {
            "title": "Test Case 3: Fail-safe State (Quantifiers, Tolerances & Timing)",
            "question": (
                "Evaluate this safety-critical requirement: 'Upon detecting a sensor failure, the system must immediately trigger "
                "a fail-safe state.' Which rules about timing, precision, and avoid-vague-quantifiers apply here?"
            )
        },
        {
            "title": "Test Case 4: Pedestrian Detection (Modal Verbs & Singularity)",
            "question": (
                "Evaluate this ADAS requirement: 'The front camera should detect pedestrians, and the system must brake when needed.' "
                "What rules does this violate regarding singularity, appropriate modal verbs (shall/should/must), and precision?"
            )
        }
    ]

    # Wrap both the input and the action buttons in a tight horizontal grid constraint
    _, center_boundary_col, _ = st.columns([1, 2.5, 1]) # Pushes layout into center 60%

    with center_boundary_col:
        st.markdown("<br>", unsafe_allow_html=True)
        user_query = st.text_input(
            "Enter a requirement to evaluate or query the knowledge base...",
            value=st.session_state.query_val,
            key="main_search_input",
            label_visibility="collapsed"
        )
        
        # Now create nested side-by-side action controls 
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            search_clicked = st.button(
                "🚀 Analyze", 
                type="primary", 
                use_container_width=True,
                key="val_run_btn"
            )
        with btn_col2:
            clear_clicked = st.button(
                "🧹 Clear", 
                use_container_width=True,
                key="val_clear_all_btn"
            )
            
        if clear_clicked:
            st.session_state.pending_query = ""
            st.session_state.retrieved_chunks = None
            st.session_state.llm_analysis = None
            st.session_state.run_analysis = False
            st.rerun()

        if search_clicked:
            if user_query.strip() == "":
                st.warning("Please enter a query first.")
            else:
                st.session_state.query_val = user_query
                st.session_state.show_val_dialog = True
                st.rerun()

        # Move Suggestions below the search bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.95rem; font-weight: 500; margin-bottom: 12px;'>💡 Quick Suggestions</div>", unsafe_allow_html=True)
        
        cols = st.columns(4)
        for idx, test in enumerate(TEST_QUESTIONS):
            with cols[idx]:
                if st.button(test["title"], key=f"btn_suggestion_{idx}", use_container_width=True):
                    st.session_state.pending_query = test["question"]
                    st.session_state.show_val_dialog = True
                    st.rerun()

    # Call the collection select dialog if state is active
    if st.session_state.get("show_val_dialog", False):
        choose_collection_dialog(st.session_state.query_val)

    # RAG Pipeline execution
    if st.session_state.run_analysis:
        query_collection = st.session_state.get("query_collection", "All Collections")
        if not nvidia or not qdrant:
            st.error("Cannot run query: NVIDIA or Qdrant client is not initialized.")
            st.session_state.run_analysis = False
        else:
            with st.spinner("Embedding query & searching Vector DB..."):
                try:
                    if query_collection == "None":
                        st.session_state.retrieved_chunks = []
                        context_block = "No guidelines retrieved (None selected)."
                    elif query_collection == "All Collections":
                        q_vec = embed_query(st.session_state.query_val)
                        collections_list = []
                        try:
                            collections_list = [c.name for c in qdrant.get_collections().collections]
                        except Exception:
                            collections_list = [DEFAULT_COLLECTION]
                            
                        all_points = []
                        for col in collections_list:
                            try:
                                search_results = qdrant.query_points(
                                    col,
                                    query=q_vec,
                                    limit=3,
                                    with_payload=True
                                )
                                for p in search_results.points:
                                    if "metadata" not in p.payload:
                                        p.payload["metadata"] = {}
                                    p.payload["metadata"]["source_collection"] = col
                                    all_points.append(p)
                            except Exception:
                                pass
                        all_points.sort(key=lambda x: x.score, reverse=True)
                        st.session_state.retrieved_chunks = all_points[:4]
                        
                        context_texts = []
                        for r in st.session_state.retrieved_chunks:
                            meta = r.payload.get("metadata", {})
                            text = r.payload.get("text", "")
                            section = meta.get("section", "N/A")
                            item_id = meta.get("item_id") or "N/A"
                            item_name = meta.get("item_name") or "N/A"
                            page = meta.get("page", "N/A")
                            source_col = meta.get("source_collection", "Unknown")
                            context_texts.append(f"Source: Collection {source_col}, Section {section}, Page {page}, Item {item_id} ({item_name})\nContent: {text}")
                        context_block = "\n\n---\n\n".join(context_texts)
                    else:
                        q_vec = embed_query(st.session_state.query_val)
                        search_results = qdrant.query_points(
                            query_collection,
                            query=q_vec,
                            limit=4,
                            with_payload=True
                        )
                        st.session_state.retrieved_chunks = search_results.points
                        
                        context_texts = []
                        for r in search_results.points:
                            meta = r.payload.get("metadata", {})
                            text = r.payload.get("text", "")
                            section = meta.get("section", "N/A")
                            item_id = meta.get("item_id") or "N/A"
                            item_name = meta.get("item_name") or "N/A"
                            page = meta.get("page", "N/A")
                            context_texts.append(f"Source: Section {section}, Page {page}, Item {item_id} ({item_name})\nContent: {text}")
                        context_block = "\n\n---\n\n".join(context_texts)
                        
                    system_prompt = (
                        "You are an expert systems engineer specializing in automotive systems and ADAS. "
                        "Use the provided context from the INCOSE Guide to Writing Requirements to answer the question. "
                        "Critique the requirement under evaluation using specific rules (R##) or characteristics (C##) "
                        "mentioned in the context. Show how to write it correctly based on the INCOSE guidance. "
                        "If the context doesn't contain the specific rule, use your general engineering knowledge to apply "
                        "INCOSE-aligned reasoning, but prioritize citing rules found in the context."
                    )
                    user_prompt = f"Context:\n{context_block}\n\nQuestion:\n{st.session_state.query_val}"
                    
                    with st.spinner("Generating compliance report..."):
                        completion = nvidia.chat.completions.create(
                            model=CHAT_MODEL,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.1,
                            max_tokens=1024
                        )
                        st.session_state.llm_analysis = completion.choices[0].message.content
                        
                except Exception as e:
                    st.error(f"Error executing RAG pipeline: {e}")
                    st.session_state.retrieved_chunks = None
                    st.session_state.llm_analysis = None
                finally:
                    st.session_state.run_analysis = False
                    st.rerun()

    # Display Results
    if st.session_state.retrieved_chunks is not None:
        st.markdown("---")
        
        # Display the Compliance Critique (the response) directly
        st.markdown("### 🤖 INCOSE Compliance Critique")
        if st.session_state.llm_analysis:
            st.markdown(st.session_state.llm_analysis)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Collapsible expander for the retrieved database chunks
        query_collection = st.session_state.get("query_collection", "All Collections")
        with st.expander(f"📚 View Retrieved Guidelines (Source: {query_collection})", expanded=False):
            if not st.session_state.retrieved_chunks:
                st.info("No guidelines retrieved (None selected or no search performed).")
            for rank, r in enumerate(st.session_state.retrieved_chunks, 1):
                meta = r.payload.get("metadata", {})
                text = r.payload.get("text", "")
                section = meta.get("section", "N/A")
                item_id = meta.get("item_id") or "N/A"
                item_name = meta.get("item_name") or "N/A"
                page = meta.get("page", "N/A")
                score = r.score
                source_col = meta.get("source_collection", query_collection)
                item_type = meta.get("item_type", "N/A")
                keywords = meta.get("keywords", [])
                
                safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
                safe_item_name = item_name.replace("<", "&lt;").replace(">", "&gt;")
                safe_item_id = item_id.replace("<", "&lt;").replace(">", "&gt;")
                
                tags_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords])
                
                card_html = f"""
                <div class="chunk-card">
                    <div class="chunk-header">
                        <span class="chunk-badge score">#{rank} • Score {score:.4f}</span>
                        <span class="chunk-badge section">Sec: {section}</span>
                        <span class="chunk-badge item">{safe_item_id}</span>
                        <span class="chunk-badge page">Page {page}</span>
                        <span class="chunk-badge type">DB: {source_col}</span>
                        <span class="chunk-badge type">{item_type}</span>
                    </div>
                    <div class="chunk-body">
                        <strong>{safe_item_name}:</strong> {safe_text}
                    </div>
                    <div class="tag-container">
                        {tags_html}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
