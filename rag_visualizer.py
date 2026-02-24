import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import re
import io
import httpx
import json
import asyncio
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# --- Configuration & App Setup ---
app = FastAPI(title="EduRAG: Pro Interactive RAG Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory State & Session Management ---
class RAGState:
    def __init__(self):
        self.raw_text = ""
        self.chunks = []
        self.vectors = None
        self.vectorizer = None
        self.pca_2d = None
        self.pca_3d = None
        self.coords_2d = []
        self.coords_3d = []
        # Optimized sliding window memory (keep last k turns)
        self.chat_history = [] 
        self.system_prompt = "You are a helpful AI assistant. Answer the user's question using ONLY the context provided below. If the answer is not in the context, say you don't know."

state = RAGState()

# --- Helpers ---

def get_favicon():
    return JSONResponse(status_code=204)

app.add_api_route("/favicon.ico", get_favicon, methods=["GET"])

# --- Advanced Text Processing ---

def extract_text_from_file(filename: str, content: bytes) -> str:
    filename = filename.lower()
    text = ""
    try:
        if filename.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(content))
                # Try to preserve some layout for tables
                text = "\n\n".join([page.extract_text(extraction_mode="layout") for page in reader.pages if page.extract_text()])
            except ImportError: return "Error: Install 'pypdf' (pip install pypdf) for PDF support."
        elif filename.endswith(".docx"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(content))
                text = "\n".join([p.text for p in doc.paragraphs])
            except ImportError: return "Error: Install 'python-docx' (pip install python-docx) for DOCX support."
        else:
            text = content.decode("utf-8")
            
        # Basic cleanup but preserve newlines for tables
        text = re.sub(r'[ \t]+', ' ', text) # Merge spaces/tabs but keep newlines
        return text.strip()
    except Exception as e:
        return f"Error processing file: {str(e)}"

def smart_chunker(text: str, method: str, chunk_size: int, overlap: int):
    """
    Advanced chunking strategies with fixed sliding window logic.
    """
    text_len = len(text)
    chunks = []
    
    if method == "section":
        # Split by double newlines (paragraphs)
        raw_splits = re.split(r'\n\s*\n', text)
        current_chunk = ""
        for split in raw_splits:
            split = split.strip()
            if not split: continue
            
            if len(current_chunk) + len(split) < chunk_size:
                current_chunk += "\n\n" + split if current_chunk else split
            else:
                if current_chunk: chunks.append(current_chunk)
                current_chunk = split
        if current_chunk: chunks.append(current_chunk)
        
    else: # "recursive" (default)
        start = 0
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # If we are not at the absolute end, try to find a natural break point
            if end < text_len:
                # Look for sentence ending
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + (chunk_size * 0.5):
                    end = last_period + 1
                else:
                    # Fallback to space
                    last_space = text.rfind(' ', start, end)
                    if last_space != -1 and last_space > start + (chunk_size * 0.3):
                        end = last_space
            
            chunk = text[start:end].strip()
            if chunk: 
                chunks.append(chunk)
            
            # Critical Fix: Break loop if we reached the end to prevent stuttering
            if end >= text_len:
                break
                
            # Calculate step size
            processed_length = end - start
            step = max(1, processed_length - overlap)
            start += step
            
    return chunks

def get_influencers(query_vec, doc_vec, feature_names, top_n=5):
    # Element-wise multiplication to find terms present in both
    contributions = query_vec * doc_vec
    top_indices = contributions.argsort()[-top_n:][::-1]
    influencers = []
    for idx in top_indices:
        score = contributions[idx]
        if score > 0.01: # Filter noise
            influencers.append({"term": feature_names[idx], "score": round(float(score), 4)})
    return influencers

# --- Core Logic ---

def update_knowledge_base(text: str, chunk_method: str = "recursive", chunk_size: int = 300, overlap: int = 50):
    state.raw_text = text
    state.chunks = smart_chunker(text, chunk_method, chunk_size, overlap)
    state.chat_history = [] 
    
    if not state.chunks: return {"status": "empty"}

    # 1. Vectorization
    state.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    state.vectors = state.vectorizer.fit_transform(state.chunks).toarray()
    
    # 2. PCA Projections
    pca2 = PCA(n_components=2)
    coords2 = pca2.fit_transform(state.vectors) if len(state.chunks) > 2 else np.random.rand(len(state.chunks), 2)
    
    pca3 = PCA(n_components=3)
    coords3 = pca3.fit_transform(state.vectors) if len(state.chunks) > 3 else np.random.rand(len(state.chunks), 3)

    state.pca_2d = pca2
    state.pca_3d = pca3
    state.coords_2d = coords2.tolist()
    state.coords_3d = coords3.tolist()

    return {
        "chunk_count": len(state.chunks),
        "coords_2d": state.coords_2d,
        "coords_3d": state.coords_3d,
        "chunks": state.chunks,
        "log": [
            f"Ingested text ({len(text)} chars)",
            f"Applied '{chunk_method}' chunking strategy",
            f"Generated {len(state.chunks)} semantic chunks",
            f"Vectorized with TF-IDF ({state.vectors.shape[1]} features)",
            "Computed PCA projections for visualization"
        ]
    }

async def call_external_llm(prompt: str, api_key: str, base_url: str, model: str):
    if not api_key:
        await asyncio.sleep(1.0) 
        return "NOTE: API Key not set. \n\nBased on the retrieved context, I can see the information you are looking for. In a real RAG system, the LLM would now synthesize the highlighted chunks above into a concise answer."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data['choices'][0]['message']['content']
    except Exception as e:
        return f"LLM API Error: {str(e)}"

# --- Models & API ---
class ConfigRequest(BaseModel):
    query: str = ""
    chunk_method: str = "recursive"
    chunk_size: int = 300
    overlap: int = 50
    top_k: int = 3
    api_key: str = ""
    base_url: str = "https://api.routeway.ai/v1" 
    model_id: str = "devstral-2512:free"

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text_from_file(file.filename, content)
    
    if text.startswith("Error"): 
        return JSONResponse(400, {"message": text, "error": True})
    
    data = update_knowledge_base(text)
    return {"message": "Processed", "data": data, "error": False}

@app.post("/api/reindex")
async def reindex(req: ConfigRequest):
    if not state.raw_text: return JSONResponse(400, {"message": "No text uploaded", "error": True})
    data = update_knowledge_base(state.raw_text, req.chunk_method, req.chunk_size, req.overlap)
    return {"message": "Reindexed", "data": data}

@app.post("/api/chat")
async def chat(req: ConfigRequest):
    if not state.vectorizer: 
        raise HTTPException(400, "Knowledge base empty. Upload a file first.")

    # 1. Embed Query
    query_vec = state.vectorizer.transform([req.query]).toarray()
    
    # 2. Retrieve & Rank
    similarities = cosine_similarity(query_vec, state.vectors).flatten()
    # Get indices sorted by score
    top_indices = similarities.argsort()[-req.top_k:][::-1]
    
    results = []
    feature_names = state.vectorizer.get_feature_names_out()
    context_text = ""
    
    # Dynamic Relevance Threshold (Simple heuristic: must be > 0.1 to be "relevant")
    relevance_threshold = 0.1
    
    top_vector = None
    
    for i, idx in enumerate(top_indices):
        idx = int(idx)
        score = float(similarities[idx])
        
        # Save top vector for inspection
        if i == 0:
            top_vector = state.vectors[idx].tolist()

        # Determine relevance
        is_relevant = score > relevance_threshold
        
        chunk_text = state.chunks[idx]
        influencers = get_influencers(query_vec[0], state.vectors[idx], feature_names)
        
        results.append({
            "index": idx,
            "score": score,
            "text": chunk_text,
            "influencers": influencers,
            "is_relevant": is_relevant
        })
        
        # Only add to context if relevant (Simulation of strict RAG)
        if is_relevant:
            context_text += f"-- Chunk {idx} (Relevance: {score:.2f}) --\n{chunk_text}\n\n"

    # 3. Memory Optimization (Sliding Window: Keep last 4 turns)
    state.chat_history = state.chat_history[-8:] 
    history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state.chat_history])
    
    # 4. Prompt Formulation
    final_prompt = f"""
[System Instruction]
{state.system_prompt}

[Chat Memory]
{history_str}

[Retrieved Context]
{context_text if context_text else "No highly relevant context found."}

[Current User Query]
{req.query}
    """.strip()

    # 5. LLM Call
    ai_response = await call_external_llm(final_prompt, req.api_key, req.base_url, req.model_id)
    
    # Update Memory
    state.chat_history.append({"role": "user", "content": req.query})
    state.chat_history.append({"role": "assistant", "content": ai_response})

    # 6. Viz Data
    q_2d = state.pca_2d.transform(query_vec)[0].tolist() if state.pca_2d else [0,0]
    q_3d = state.pca_3d.transform(query_vec)[0].tolist() if state.pca_3d else [0,0,0]

    return {
        "answer": ai_response,
        "retrieved_chunks": results,
        "query_coords_2d": q_2d,
        "query_coords_3d": q_3d,
        "glass_box": {
            "formulated_prompt": final_prompt,
            "raw_response": ai_response,
            "relevance_threshold": relevance_threshold,
            "system_prompt": state.system_prompt,
            "history": state.chat_history[:-2] # Return history excluding current turn for display
        },
        "vectors": {
            "query": query_vec[0].tolist(), # Send first 100 dims or handled in UI? TFIDF is sparse but array is dense.
            "top_chunk": top_vector
        },
        "process_log": [
            "Vectorized user query",
            f"Calculated similarity against {len(state.chunks)} chunks",
            f"Identified {len([r for r in results if r['is_relevant']])} relevant chunks (> {relevance_threshold})",
            "Optimized memory (sliding window)",
            "Dispatched structured prompt to LLM"
        ]
    }

# --- Frontend ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EduRAG Pro: Glass Box Edition</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.0.2/marked.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        
        .chunk-highlight { transition: all 0.2s; cursor: pointer; border-radius: 2px; }
        /* Retrieved Chunk Style */
        .chunk-retrieved { background-color: rgba(99, 102, 241, 0.25); border-bottom: 2px solid #6366f1; }
        /* Active (Hovered/Clicked) Chunk Style */
        .chunk-active { background-color: rgba(250, 204, 21, 0.5); border-bottom: 2px solid #ca8a04; transform: scale(1.02); display: inline-block; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #f1f5f9; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        
        .tooltip-trigger:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #1e293b;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 10px;
            white-space: nowrap;
            z-index: 50;
            pointer-events: none;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        @keyframes pulse-ring { 0% { transform: scale(0.8); opacity: 0.5; } 100% { transform: scale(2); opacity: 0; } }
        .pulse-dot::before { content: ''; position: absolute; width: 100%; height: 100%; background-color: inherit; border-radius: 50%; z-index: -1; animation: pulse-ring 1.5s cubic-bezier(0.215, 0.61, 0.355, 1) infinite; }
        
        .flow-step { transition: all 0.3s ease; }
        .flow-step:hover { transform: translateY(-2px); }
        .flow-active { border-color: #6366f1; background-color: #eef2ff; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15); }
        .flow-line { transition: all 0.5s ease; }
        .flow-line.active { background-color: #6366f1; }
    </style>
</head>
<body class="bg-slate-100 text-slate-800 h-screen overflow-hidden flex flex-col">

    <!-- Header -->
    <header class="bg-white border-b border-slate-200 h-14 flex items-center justify-between px-4 shrink-0 shadow-sm z-20">
        <div class="flex items-center gap-2">
            <div class="bg-indigo-600 text-white w-8 h-8 flex items-center justify-center rounded-lg shadow-sm">
                <i class="fa-solid fa-layer-group"></i>
            </div>
            <div>
                <h1 class="font-bold text-slate-800 leading-tight">EduRAG <span class="text-indigo-600">GlassBox</span></h1>
                <p class="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Educational RAG Architecture</p>
            </div>
        </div>
        
        <div class="flex gap-4 items-center">
            <button onclick="toggleLog()" class="text-xs font-semibold text-slate-500 hover:text-indigo-600 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200 transition">
                <i class="fa-solid fa-terminal mr-1"></i> Logs
            </button>
            
            <div class="bg-slate-100 p-1 rounded-lg flex text-xs font-semibold">
                <button onclick="setMode('2d')" id="btn2d" class="px-3 py-1 rounded-md bg-white shadow text-indigo-600 transition" data-tooltip="Visualize vectors in 2D plane">2D View</button>
                <button onclick="setMode('3d')" id="btn3d" class="px-3 py-1 rounded-md text-slate-500 hover:text-slate-700 transition" data-tooltip="Visualize vectors in 3D space">3D View</button>
            </div>
            
            <button onclick="document.getElementById('fileInput').click()" class="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-1.5 rounded-lg text-sm font-medium transition shadow-sm flex items-center gap-2">
                <i class="fa-solid fa-upload"></i> Upload Data
            </button>
            <input type="file" id="fileInput" class="hidden" accept=".txt,.md,.pdf,.docx">
        </div>
    </header>

    <!-- Main Workspace -->
    <div class="flex flex-1 overflow-hidden">
        
        <!-- Left: Document Corpus -->
        <div class="w-[28%] bg-white border-r border-slate-200 flex flex-col shadow-sm z-10">
            <div class="p-3 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center backdrop-blur">
                <div class="text-xs font-bold text-slate-400 uppercase tracking-wider tooltip-trigger relative" data-tooltip="Raw text content split into chunks">
                    <i class="fa-regular fa-file-alt mr-1"></i> Knowledge Base
                </div>
                <span id="docStats" class="text-[10px] bg-slate-200 px-2 py-0.5 rounded-full text-slate-600">Empty</span>
            </div>
            <!-- Document View -->
            <div id="documentView" class="flex-1 overflow-y-auto p-6 text-sm leading-7 font-serif text-slate-600 selection:bg-indigo-100 whitespace-pre-wrap">
                <div class="h-full flex flex-col items-center justify-center text-slate-400 opacity-60">
                    <i class="fa-solid fa-cloud-arrow-up text-4xl mb-3"></i>
                    <p class="text-center text-xs mt-2">Upload a document to begin.<br>Supports PDF, DOCX, TXT.</p>
                </div>
            </div>
            <div class="p-2 bg-yellow-50 border-t border-yellow-100 text-[10px] text-yellow-700 flex items-center gap-2">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <span><strong>Note:</strong> Images are skipped in this Text-RAG demo.</span>
            </div>
        </div>

        <!-- Middle: Visualization & Pipeline -->
        <div class="flex-1 flex flex-col bg-slate-50 relative">
            
            <!-- Visualization Canvas -->
            <div class="flex-1 relative">
                <div id="plotDiv" class="w-full h-full"></div>
                
                <!-- Floating Info Card -->
                <div id="nodeInfo" class="hidden absolute top-4 left-4 bg-white/90 backdrop-blur border border-slate-200 p-3 rounded-lg shadow-lg w-64 text-xs z-30 pointer-events-none">
                    <h4 class="font-bold text-indigo-600 mb-1">Chunk #<span id="infoId"></span></h4>
                    <p id="infoText" class="text-slate-600 line-clamp-3 mb-2 italic"></p>
                </div>

                <!-- Live Process Log Overlay -->
                <div id="processOverlay" class="hidden absolute bottom-4 left-4 right-4 z-20 pointer-events-none">
                    <div class="bg-white/95 backdrop-blur border border-indigo-100 rounded-xl p-4 shadow-xl max-w-lg mx-auto pointer-events-auto">
                         <h4 class="text-xs font-bold text-indigo-600 uppercase mb-2 flex items-center justify-between">
                            <span class="flex items-center"><span class="w-2 h-2 rounded-full bg-indigo-600 mr-2 pulse-dot relative"></span> Execution Log</span>
                            <span class="text-[9px] text-slate-400 cursor-pointer hover:text-red-500" onclick="toggleLog()">Close</span>
                        </h4>
                        <div id="processLog" class="text-xs font-mono text-slate-600 space-y-1 max-h-32 overflow-y-auto"></div>
                    </div>
                </div>
            </div>

            <!-- Glass Box / Deep Dive Inspector -->
            <div class="h-[45%] bg-white border-t border-slate-200 flex flex-col shrink-0 shadow-[0_-4px_24px_rgba(0,0,0,0.03)] z-20">
                <!-- Tabs -->
                <div class="flex border-b border-slate-100 overflow-x-auto">
                    <button onclick="showGlassTab('pipeline')" class="glass-tab px-4 py-3 text-xs font-bold text-indigo-600 border-b-2 border-indigo-600 hover:bg-indigo-50 transition whitespace-nowrap">Interactive Pipeline</button>
                    <button onclick="showGlassTab('analysis')" class="glass-tab px-4 py-3 text-xs font-bold text-slate-500 hover:text-indigo-600 hover:bg-slate-50 transition whitespace-nowrap">Analysis & Influencers</button>
                    <button onclick="showGlassTab('memory')" class="glass-tab px-4 py-3 text-xs font-bold text-slate-500 hover:text-indigo-600 hover:bg-slate-50 transition whitespace-nowrap">Memory & Context</button>
                    <button onclick="showGlassTab('vectors')" class="glass-tab px-4 py-3 text-xs font-bold text-slate-500 hover:text-indigo-600 hover:bg-slate-50 transition whitespace-nowrap">Raw Vectors</button>
                    <button onclick="showGlassTab('prompt')" class="glass-tab px-4 py-3 text-xs font-bold text-slate-500 hover:text-indigo-600 hover:bg-slate-50 transition whitespace-nowrap">LLM Prompt</button>
                </div>
                
                <!-- Tab Content -->
                <div class="flex-1 overflow-hidden relative bg-slate-50/50">
                    
                    <!-- 1. Pipeline Tab (Interactive Flow) -->
                    <div id="glass-pipeline" class="glass-pane absolute inset-0 p-6 overflow-y-auto flex flex-col items-center justify-center">
                        <div class="flex items-center w-full max-w-2xl justify-between">
                            <!-- Steps -->
                            <div id="step-ingest" class="flow-step group relative flex flex-col items-center cursor-pointer" onclick="showStepInfo('ingest')">
                                <div class="w-12 h-12 rounded-xl bg-white border-2 border-slate-200 flex items-center justify-center text-slate-500 mb-2 shadow-sm group-hover:border-indigo-400 group-hover:text-indigo-600 transition">
                                    <i class="fa-solid fa-file-lines text-lg"></i>
                                </div>
                                <div class="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Ingest</div>
                            </div>
                            <div class="flow-line h-0.5 bg-slate-200 flex-1 mx-2"></div>
                            
                            <div id="step-chunk" class="flow-step group relative flex flex-col items-center cursor-pointer" onclick="showStepInfo('chunk')">
                                <div class="w-12 h-12 rounded-xl bg-white border-2 border-slate-200 flex items-center justify-center text-slate-500 mb-2 shadow-sm group-hover:border-indigo-400 group-hover:text-indigo-600 transition">
                                    <i class="fa-solid fa-scissors text-lg"></i>
                                </div>
                                <div class="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Chunking</div>
                            </div>
                            <div class="flow-line h-0.5 bg-slate-200 flex-1 mx-2"></div>

                            <div id="step-embed" class="flow-step group relative flex flex-col items-center cursor-pointer" onclick="showStepInfo('embed')">
                                <div class="w-12 h-12 rounded-xl bg-white border-2 border-slate-200 flex items-center justify-center text-slate-500 mb-2 shadow-sm group-hover:border-indigo-400 group-hover:text-indigo-600 transition">
                                    <i class="fa-solid fa-cube text-lg"></i>
                                </div>
                                <div class="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Embedding</div>
                            </div>
                            <div class="flow-line h-0.5 bg-slate-200 flex-1 mx-2"></div>

                            <div id="step-retrieve" class="flow-step group relative flex flex-col items-center cursor-pointer" onclick="showStepInfo('retrieve')">
                                <div class="w-12 h-12 rounded-xl bg-white border-2 border-slate-200 flex items-center justify-center text-slate-500 mb-2 shadow-sm group-hover:border-indigo-400 group-hover:text-indigo-600 transition">
                                    <i class="fa-solid fa-magnifying-glass text-lg"></i>
                                </div>
                                <div class="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Retrieval</div>
                            </div>
                            <div class="flow-line h-0.5 bg-slate-200 flex-1 mx-2"></div>

                            <div id="step-generate" class="flow-step group relative flex flex-col items-center cursor-pointer" onclick="showStepInfo('generate')">
                                <div class="w-12 h-12 rounded-xl bg-white border-2 border-slate-200 flex items-center justify-center text-slate-500 mb-2 shadow-sm group-hover:border-indigo-400 group-hover:text-indigo-600 transition">
                                    <i class="fa-solid fa-wand-magic-sparkles text-lg"></i>
                                </div>
                                <div class="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Generation</div>
                            </div>
                        </div>
                        
                        <div id="stepInfoBox" class="mt-6 p-4 bg-white border border-indigo-100 rounded-lg shadow-sm max-w-2xl w-full text-xs text-slate-600 hidden">
                            <!-- Dynamic Content -->
                        </div>
                        <div class="mt-4 text-[10px] text-slate-400 italic">Click any step above to understand how it works.</div>
                    </div>

                    <!-- 2. Analysis Tab -->
                    <div id="glass-analysis" class="glass-pane absolute inset-0 p-4 overflow-y-auto hidden flex gap-4">
                        <div class="w-1/2 border-r border-slate-200 pr-4">
                            <h5 class="text-[10px] font-bold text-slate-500 uppercase mb-3">Relevance Scoring (Cosine Similarity)</h5>
                            <div id="relevanceBars" class="space-y-3">
                                <div class="text-center text-slate-400 text-xs mt-10 italic">Run a query to see relevance scores</div>
                            </div>
                        </div>
                        <div class="w-1/2 pl-4">
                            <h5 class="text-[10px] font-bold text-slate-500 uppercase mb-3">Influencer Keywords (Why this chunk?)</h5>
                            <div id="influencerTags" class="flex flex-wrap gap-2"></div>
                            <p class="text-[10px] text-slate-400 mt-2">These specific terms in the chunk had the highest mathematical overlap with your query vector.</p>
                        </div>
                    </div>

                    <!-- 3. Memory Tab -->
                    <div id="glass-memory" class="glass-pane absolute inset-0 p-0 overflow-y-auto hidden flex">
                        <div class="w-1/3 border-r border-slate-200 p-4 bg-slate-50">
                            <h5 class="text-[10px] font-bold text-purple-600 uppercase mb-2">System Instruction</h5>
                            <div id="memSystem" class="text-[10px] font-mono text-slate-600 bg-white p-2 rounded border border-slate-200"></div>
                        </div>
                        <div class="w-1/3 border-r border-slate-200 p-4 bg-white">
                            <h5 class="text-[10px] font-bold text-blue-600 uppercase mb-2">Short-Term History</h5>
                            <div id="memHistory" class="space-y-2"></div>
                        </div>
                        <div class="w-1/3 p-4 bg-yellow-50/30">
                            <h5 class="text-[10px] font-bold text-amber-600 uppercase mb-2">Retrieved Context Window</h5>
                             <div id="memContext" class="space-y-2"></div>
                        </div>
                    </div>

                    <!-- 4. Vectors Tab -->
                    <div id="glass-vectors" class="glass-pane absolute inset-0 p-4 overflow-y-auto hidden">
                        <div class="grid grid-cols-2 gap-4 h-full">
                            <div class="flex flex-col">
                                <h5 class="text-[10px] font-bold text-slate-500 uppercase mb-2">Query Vector (First 50 dims)</h5>
                                <div id="vecQuery" class="flex-1 bg-slate-900 text-green-400 font-mono text-[9px] p-3 rounded overflow-auto whitespace-pre-wrap">Waiting for query...</div>
                            </div>
                            <div class="flex flex-col">
                                <h5 class="text-[10px] font-bold text-slate-500 uppercase mb-2">Top Chunk Vector (First 50 dims)</h5>
                                <div id="vecChunk" class="flex-1 bg-slate-900 text-blue-400 font-mono text-[9px] p-3 rounded overflow-auto whitespace-pre-wrap">Waiting for retrieval...</div>
                            </div>
                        </div>
                    </div>

                    <!-- 5. Prompt Tab -->
                    <div id="glass-prompt" class="glass-pane absolute inset-0 p-4 overflow-y-auto hidden">
                        <h5 class="text-[10px] font-bold text-slate-500 uppercase mb-2">Final Assembled Prompt</h5>
                        <pre id="promptPreview" class="text-[10px] font-mono bg-white p-3 rounded border border-slate-200 whitespace-pre-wrap text-slate-600 shadow-sm"></pre>
                        
                        <h5 class="text-[10px] font-bold text-slate-500 uppercase mb-2 mt-4">Raw API Response</h5>
                        <pre id="responsePreview" class="text-[10px] font-mono bg-white p-3 rounded border border-slate-200 whitespace-pre-wrap text-slate-600 shadow-sm"></pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right: Interface -->
        <div class="w-[28%] bg-white border-l border-slate-200 flex flex-col shadow-sm z-10">
            <!-- Tabs -->
            <div class="flex border-b border-slate-200 text-xs font-semibold text-slate-500">
                <button onclick="switchTab('chat')" id="tab-chat" class="flex-1 py-3 text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50/50">Chat & Query</button>
                <button onclick="switchTab('settings')" id="tab-settings" class="flex-1 py-3 hover:text-slate-700 hover:bg-slate-50">Settings</button>
            </div>

            <!-- Settings -->
            <div id="view-settings" class="hidden p-5 space-y-6 overflow-y-auto bg-slate-50 h-full">
                <div class="space-y-4">
                    <h3 class="text-xs font-bold text-slate-400 uppercase tracking-wider">LLM Provider</h3>
                    <div class="bg-white p-3 rounded border border-slate-200 shadow-sm">
                        <label class="block text-xs font-semibold text-slate-700 mb-1">API Key</label>
                        <input type="password" id="apiKey" placeholder="sk-..." class="w-full text-xs border border-slate-200 rounded p-2 mb-2 focus:border-indigo-500 focus:outline-none">
                        <label class="block text-xs font-semibold text-slate-700 mb-1">Base URL</label>
                        <input type="text" id="baseUrl" value="https://api.routeway.ai/v1" class="w-full text-xs border border-slate-200 rounded p-2 mb-2 focus:border-indigo-500 focus:outline-none">
                        <label class="block text-xs font-semibold text-slate-700 mb-1">Model ID</label>
                        <input type="text" id="modelId" value="devstral-2512:free" class="w-full text-xs border border-slate-200 rounded p-2 focus:border-indigo-500 focus:outline-none">
                    </div>

                    <h3 class="text-xs font-bold text-slate-400 uppercase tracking-wider">Chunking Strategy</h3>
                    <div class="bg-white p-3 rounded border border-slate-200 shadow-sm">
                        <label class="block text-xs font-semibold text-slate-700 mb-2">Method</label>
                        <select id="chunkMethod" class="w-full text-xs border border-slate-200 rounded p-2 bg-slate-50 mb-3">
                            <option value="recursive">Recursive Character (Best for Text)</option>
                            <option value="section">Paragraph/Section (Best for Structure)</option>
                        </select>
                        <label class="block text-xs font-semibold text-slate-700 mb-1">Chunk Size: <span id="chunkSizeVal">300</span></label>
                        <input type="range" id="chunkSize" min="100" max="1000" step="50" value="300" class="w-full accent-indigo-600">
                    </div>
                </div>
                <button onclick="triggerReindex()" class="w-full py-2 bg-slate-800 text-white text-xs font-bold rounded shadow hover:bg-slate-700 transition">Apply & Re-Index</button>
            </div>

            <!-- Chat -->
            <div id="view-chat" class="flex flex-col h-full">
                <div id="chatHistory" class="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50">
                    <div class="bg-indigo-50 border border-indigo-100 p-3 rounded-lg rounded-tl-none text-xs text-indigo-800 shadow-sm">
                        <p class="font-bold mb-1"><i class="fa-solid fa-robot mr-1"></i> RAG Ready</p>
                        <p>Ask a question. I will show you which chunks are retrieved, why they are relevant, and how the LLM uses them.</p>
                    </div>
                </div>

                <div class="p-4 bg-white border-t border-slate-200">
                    <form id="chatForm" class="relative">
                        <input type="text" id="userQuery" placeholder="Enter query..." 
                               class="w-full bg-slate-100 border-0 rounded-xl py-3 pl-4 pr-12 text-sm focus:ring-2 focus:ring-indigo-500 transition shadow-inner">
                        <button type="submit" id="sendBtn" class="absolute right-2 top-2 p-1.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 shadow-sm transition">
                            <i class="fa-solid fa-paper-plane text-xs"></i>
                        </button>
                    </form>
                </div>
            </div>
        </div>
    </div>

    <script>
        let mode = '3d'; // Default to 3D for impact
        let chunks = [];
        let coords2d = [];
        let coords3d = [];
        let lastRetrieved = [];
        
        // --- Initialization ---
        function initPlotly() {
            const layout = {
                margin: { l: 0, r: 0, b: 0, t: 0 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                showlegend: false,
                hovermode: 'closest',
                scene: {
                    xaxis: { title: '', showgrid: true, zeroline: false, showticklabels: false, showbackground: false },
                    yaxis: { title: '', showgrid: true, zeroline: false, showticklabels: false, showbackground: false },
                    zaxis: { title: '', showgrid: true, zeroline: false, showticklabels: false, showbackground: false },
                    bgcolor: 'rgba(0,0,0,0)',
                    camera: { eye: {x: 1.2, y: 1.2, z: 0.8}, center: {x:0, y:0, z:0} },
                    dragmode: 'orbit'
                },
                xaxis: { showgrid: false, zeroline: false, showticklabels: false },
                yaxis: { showgrid: false, zeroline: false, showticklabels: false }
            };
            Plotly.newPlot('plotDiv', [], layout, {displayModeBar: false, responsive: true});
            
            document.getElementById('plotDiv').on('plotly_hover', function(data){
                const pt = data.points[0];
                if(pt.curveNumber === 0) { 
                    showTooltip(pt.pointIndex);
                }
            });
        }
        initPlotly();

        // --- Core Functions ---

        async function logProcess(msg, delay=300) {
            const log = document.getElementById('processLog');
            if(log.innerHTML === "") log.innerHTML = ""; 
            const entry = document.createElement('div');
            entry.innerHTML = `<span class="text-indigo-400 mr-1">></span> ${msg}`;
            entry.className = "opacity-0 transform translate-x-2 transition-all duration-300";
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
            requestAnimationFrame(() => entry.classList.remove('opacity-0', 'translate-x-2'));
            await new Promise(r => setTimeout(r, delay));
        }

        // Highlight Flow Step
        async function activateStep(stepName, duration=1000) {
            // Reset
            document.querySelectorAll('.flow-step').forEach(el => el.classList.remove('flow-active'));
            document.querySelectorAll('.flow-line').forEach(el => el.classList.remove('active'));
            
            // Activate
            const stepEl = document.getElementById(`step-${stepName}`);
            if(stepEl) {
                stepEl.classList.add('flow-active');
                // Highlight line before it (simplified)
                if(stepName === 'chunk') document.querySelectorAll('.flow-line')[0].classList.add('active');
                if(stepName === 'embed') document.querySelectorAll('.flow-line')[1].classList.add('active');
            }
            await new Promise(r => setTimeout(r, duration));
        }

        function renderGraph(queryCoords = null) {
            const is3d = mode === '3d';
            const dataCoords = is3d ? coords3d : coords2d;
            if(!dataCoords.length) return;

            const x = dataCoords.map(c => c[0]);
            const y = dataCoords.map(c => c[1]);
            const z = is3d ? dataCoords.map(c => c[2]) : null;

            // Colors: Interactive
            const colors = chunks.map((_, i) => {
                const retrieved = lastRetrieved.find(r => r.index === i);
                if (retrieved) return retrieved.is_relevant ? '#6366f1' : '#f59e0b';
                return '#cbd5e1';
            });
            
            const sizes = chunks.map((_, i) => lastRetrieved.some(r => r.index === i) ? 14 : 7);
            const opacities = chunks.map((_, i) => lastRetrieved.some(r => r.index === i) ? 1 : 0.6);

            const traceDocs = {
                x: x, y: y, z: z,
                mode: 'markers',
                type: is3d ? 'scatter3d' : 'scatter',
                marker: { size: sizes, color: colors, opacity: opacities, line: {color: 'white', width: 0.5} },
                text: chunks.map((c, i) => `Chunk ${i}`),
                hoverinfo: 'text'
            };

            const traces = [traceDocs];

            if (queryCoords) {
                traces.push({
                    x: [queryCoords[0]], 
                    y: [queryCoords[1]], 
                    z: is3d ? [queryCoords[2]] : null,
                    mode: 'markers',
                    type: is3d ? 'scatter3d' : 'scatter',
                    marker: { size: 18, color: '#ef4444', symbol: is3d ? 'diamond' : 'star', line: {color: 'white', width: 2} },
                    name: 'User Query',
                    hoverinfo: 'name'
                });
            }

            const layout = {
                dragmode: is3d ? 'orbit' : 'pan',
                transition: { duration: 800, easing: 'cubic-in-out' },
                scene: { camera: { eye: {x: 1.2, y: 1.2, z: 0.8} } }
            };

            Plotly.react('plotDiv', traces, layout);
        }

        // --- Interaction Handlers ---
        
        document.getElementById('fileInput').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if(!file) return;
            
            const logDiv = document.getElementById('processLog');
            document.getElementById('processOverlay').classList.remove('hidden');
            logDiv.innerHTML = '';
            
            showGlassTab('pipeline');
            await activateStep('ingest', 500);
            await logProcess(`Uploading ${file.name}...`);
            
            const fd = new FormData();
            fd.append('file', file);
            
            try {
                const res = await fetch('/api/upload', {method: 'POST', body: fd});
                const data = await res.json();
                
                if(data.error) {
                    await logProcess(`ERROR: ${data.message}`);
                    return;
                }
                
                await activateStep('chunk', 800);
                await activateStep('embed', 800);
                
                for(const log of data.data.log) await logProcess(log, 150);
                
                handleDataUpdate(data.data);
                addChatMsg("system", `Processed ${file.name} into ${data.data.chunk_count} chunks.`);
            } catch(e) { 
                console.error(e);
                await logProcess("Upload Failed.");
            }
        });

        document.getElementById('chatForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const q = document.getElementById('userQuery').value;
            if(!q) return;
            
            document.getElementById('userQuery').value = "";
            addChatMsg("user", q);
            
            const logDiv = document.getElementById('processLog');
            document.getElementById('processOverlay').classList.remove('hidden');
            logDiv.innerHTML = ''; 
            
            showGlassTab('pipeline');

            const payload = {
                query: q,
                chunk_method: document.getElementById('chunkMethod').value,
                chunk_size: parseInt(document.getElementById('chunkSize').value),
                overlap: 50,
                api_key: document.getElementById('apiKey').value,
                base_url: document.getElementById('baseUrl').value,
                model_id: document.getElementById('modelId').value
            };

            try {
                // Visualize Pipeline
                await activateStep('embed', 600);
                await logProcess("Embedding Query...", 100);
                
                const res = await fetch('/api/chat', {
                    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
                });
                const data = await res.json();
                
                await activateStep('retrieve', 600);
                for(const step of data.process_log) await logProcess(step, 150);

                // Update Visuals
                lastRetrieved = data.retrieved_chunks;
                renderGraph(mode === '3d' ? data.query_coords_3d : data.query_coords_2d);
                
                // Update Glass Box
                updateGlassBox(data.retrieved_chunks, data.glass_box, data.vectors);
                
                // Response
                await activateStep('generate', 600);
                addChatMsg("ai", data.answer);
                
                highlightChunksInText(data.retrieved_chunks);
                
                // Auto switch to Analysis after a moment
                setTimeout(() => showGlassTab('analysis'), 2000);

            } catch(e) {
                console.error(e);
                addChatMsg("system", "Error occurred. Check logs.");
            }
        });

        // --- Helper Logic ---

        function handleDataUpdate(data) {
            chunks = data.chunks;
            coords2d = data.coords_2d;
            coords3d = data.coords_3d;
            lastRetrieved = [];
            
            const view = document.getElementById('documentView');
            view.innerHTML = '';
            chunks.forEach((text, i) => {
                const span = document.createElement('span');
                span.innerText = text + " "; 
                span.id = `chunk-${i}`;
                span.className = "chunk-highlight hover:bg-yellow-50";
                span.onclick = () => { showTooltip(i); highlightSingleChunk(i); };
                view.appendChild(span);
            });
            document.getElementById('docStats').innerText = `${chunks.length} Chunks`;
            renderGraph();
        }

        function highlightChunksInText(retrieved) {
            document.querySelectorAll('.chunk-retrieved').forEach(el => el.classList.remove('chunk-retrieved', 'chunk-active'));
            if(retrieved && retrieved.length > 0) {
                retrieved.forEach(r => {
                    const el = document.getElementById(`chunk-${r.index}`);
                    if(el) {
                        el.classList.add('chunk-retrieved');
                        if(r.is_relevant) el.classList.add('chunk-active');
                    }
                });
                const first = document.getElementById(`chunk-${retrieved[0].index}`);
                if(first) first.scrollIntoView({behavior: 'smooth', block: 'center'});
            }
        }
        
        function highlightSingleChunk(idx) {
             document.querySelectorAll('.chunk-active').forEach(el => el.classList.remove('chunk-active'));
             const el = document.getElementById(`chunk-${idx}`);
             if(el) el.classList.add('chunk-active');
        }

        function updateGlassBox(results, glassData, vectors) {
            // 1. Relevance Bars
            const barsContainer = document.getElementById('relevanceBars');
            barsContainer.innerHTML = '';
            const threshold = glassData.relevance_threshold;
            
            results.forEach(r => {
                const width = Math.min(100, r.score * 100);
                const colorClass = r.score > threshold ? 'bg-indigo-600' : 'bg-yellow-400';
                barsContainer.innerHTML += `
                    <div>
                        <div class="flex justify-between text-[10px] mb-1">
                            <span class="font-bold">Chunk ${r.index}</span>
                            <span>${r.score.toFixed(3)} ${r.score < threshold ? '(Low)' : ''}</span>
                        </div>
                        <div class="w-full bg-slate-100 rounded-full h-2 relative">
                            <div class="${colorClass} h-2 rounded-full" style="width: ${width}%"></div>
                            <div class="absolute top-0 bottom-0 w-0.5 bg-red-300 z-10" style="left: ${threshold * 100}%" title="Threshold: ${threshold}"></div>
                        </div>
                    </div>
                `;
            });

            // 2. Influencers
            const tagsContainer = document.getElementById('influencerTags');
            tagsContainer.innerHTML = '';
            const allInfluencers = results.flatMap(r => r.influencers);
            const uniqueTerms = [...new Set(allInfluencers.map(i => i.term))];
            uniqueTerms.slice(0, 15).forEach(term => {
                tagsContainer.innerHTML += `<span class="px-2 py-1 bg-indigo-50 text-indigo-700 text-[10px] rounded border border-indigo-100">${term}</span>`;
            });

            // 3. Memory
            document.getElementById('memSystem').innerText = glassData.system_prompt;
            const histDiv = document.getElementById('memHistory');
            histDiv.innerHTML = glassData.history.map(h => 
                `<div class="text-[10px] p-2 rounded ${h.role === 'user' ? 'bg-blue-50' : 'bg-slate-100'}">
                    <span class="font-bold uppercase text-[9px] text-slate-500">${h.role}</span>
                    <div class="line-clamp-2">${h.content}</div>
                </div>`
            ).join('');
            
            const ctxDiv = document.getElementById('memContext');
            ctxDiv.innerHTML = results.filter(r => r.is_relevant).map(r => 
                `<div class="text-[10px] p-2 bg-white border border-slate-200 rounded">
                    <span class="font-bold text-indigo-600">Chunk ${r.index}</span>
                    <div class="line-clamp-3 italic text-slate-500">${r.text}</div>
                </div>`
            ).join('') || '<div class="text-[10px] text-slate-400 italic">No context retrieved.</div>';

            // 4. Vectors
            const formatVec = (v) => v ? "[" + v.slice(0, 20).map(n => n.toFixed(3)).join(", ") + ", ...]" : "N/A";
            document.getElementById('vecQuery').innerText = formatVec(vectors.query);
            document.getElementById('vecChunk').innerText = formatVec(vectors.top_chunk);

            // 5. Prompt
            document.getElementById('promptPreview').innerText = glassData.formulated_prompt;
            document.getElementById('responsePreview').innerText = glassData.raw_response;
        }

        // --- UI Toggles ---
        function showGlassTab(tabName) {
            document.querySelectorAll('.glass-pane').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.glass-tab').forEach(el => {
                el.classList.remove('text-indigo-600', 'border-b-2', 'border-indigo-600');
                el.classList.add('text-slate-500');
            });
            
            document.getElementById(`glass-${tabName}`).classList.remove('hidden');
            // Find button by text content is hard, so we just reset all and rely on click style. 
            // In a real app we'd bind IDs. For now, simple active class logic on the button itself would be better.
            event.target.classList.remove('text-slate-500');
            event.target.classList.add('text-indigo-600', 'border-b-2', 'border-indigo-600');
        }

        function showStepInfo(step) {
            const infoBox = document.getElementById('stepInfoBox');
            infoBox.classList.remove('hidden');
            let title = "", desc = "";
            
            if(step === 'ingest') { title = "Data Ingestion"; desc = "Converting raw files (PDF, DOCX) into clean plain text strings. We also clean up whitespace and formatting here."; }
            if(step === 'chunk') { title = "Chunking Strategy"; desc = "Splitting long text into smaller, meaningful segments (e.g., 300 characters). Overlap ensures we don't cut sentences in half."; }
            if(step === 'embed') { title = "Vector Embedding"; desc = "Converting text into lists of numbers (Vectors). Similar concepts (e.g., 'dog' and 'puppy') get similar numbers."; }
            if(step === 'retrieve') { title = "Cosine Similarity Search"; desc = "Comparing the Query Vector angle against all Chunk Vectors. Small angle = High Similarity. We pick the top matches."; }
            if(step === 'generate') { title = "LLM Generation"; desc = "The retrieved text chunks are pasted into the System Prompt. The LLM then answers 'using the context below'."; }
            
            infoBox.innerHTML = `<h5 class="font-bold text-indigo-600 mb-1">${title}</h5><p>${desc}</p>`;
        }

        function showTooltip(idx) {
             document.getElementById('infoId').innerText = idx;
             document.getElementById('infoText').innerText = chunks[idx].substring(0, 150) + "...";
             document.getElementById('nodeInfo').classList.remove('hidden');
        }

        function toggleLog() {
            document.getElementById('processOverlay').classList.toggle('hidden');
        }

        function addChatMsg(role, text) {
            const h = document.getElementById('chatHistory');
            const div = document.createElement('div');
            
            if(role === 'user') {
                div.className = "bg-white border border-slate-200 p-3 rounded-lg rounded-tr-none text-xs text-slate-700 shadow-sm ml-8";
                div.innerHTML = text;
            } else if (role === 'system') {
                div.className = "text-center text-[10px] text-slate-400 my-2 italic";
                div.innerHTML = text;
            } else {
                div.className = "bg-indigo-600 text-white p-3 rounded-lg rounded-tl-none text-xs shadow-md mr-4";
                div.innerHTML = marked.parse(text);
                div.querySelectorAll('p').forEach(p => p.classList.add('mb-2'));
            }
            h.appendChild(div);
            h.scrollTop = h.scrollHeight;
        }

        function setMode(m) {
            mode = m;
            document.getElementById('btn2d').className = m === '2d' ? "px-3 py-1 rounded-md bg-white shadow text-indigo-600 transition" : "px-3 py-1 rounded-md text-slate-500 hover:text-slate-700 transition";
            document.getElementById('btn3d').className = m === '3d' ? "px-3 py-1 rounded-md bg-white shadow text-indigo-600 transition" : "px-3 py-1 rounded-md text-slate-500 hover:text-slate-700 transition";
            renderGraph(mode === '3d' ? coords3d : coords2d);
        }

        function switchTab(t) {
            document.getElementById('view-chat').classList.add('hidden');
            document.getElementById('view-settings').classList.add('hidden');
            document.getElementById(`view-${t}`).classList.remove('hidden');
            
            document.getElementById('tab-chat').className = t === 'chat' ? "flex-1 py-3 text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50/50" : "flex-1 py-3 hover:text-slate-700 hover:bg-slate-50";
            document.getElementById('tab-settings').className = t === 'settings' ? "flex-1 py-3 text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50/50" : "flex-1 py-3 hover:text-slate-700 hover:bg-slate-50";
        }
        
        async function triggerReindex() {
             await logProcess("Re-indexing...", 100);
             const payload = {
                query: "",
                chunk_method: document.getElementById('chunkMethod').value,
                chunk_size: parseInt(document.getElementById('chunkSize').value),
                overlap: 50
            };
            const res = await fetch('/api/reindex', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
            const data = await res.json();
            handleDataUpdate(data.data);
            await logProcess("Re-indexing Complete.");
            switchTab('chat');
        }
        
        document.getElementById('chunkSize').addEventListener('input', (e) => document.getElementById('chunkSizeVal').innerText = e.target.value);

    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return html_content

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)