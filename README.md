# 🧬 EduRAG: Pro Interactive RAG Visualizer

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**EduRAG GlassBox** is an advanced educational workspace designed to demystify the Retrieval-Augmented Generation (RAG) pipeline. By combining real-time vector visualization with a "Glass Box" inspector, it allows developers and students to see exactly how text is chunked, vectorized, and retrieved to inform LLM responses.



---

## ✨ Key Features

* **🔍 2D/3D Vector Space Visualization:** Watch your document chunks and user queries interact in a high-dimensional space reduced via PCA (Principal Component Analysis).
* **🛠️ Dynamic Chunking Strategies:** Toggle between `Recursive Character` and `Section-based` chunking to see how document structure affects retrieval.
* **🧠 Glass Box Inspector:** * **Relevance Scoring:** Real-time cosine similarity bars with adjustable thresholds.
    * **Influencer Analysis:** Identifies the specific keywords (TF-IDF features) that drove the retrieval.
    * **Prompt Assembly:** Peek at the final prompt sent to the LLM, including system instructions and retrieved context.
* **📂 Multi-Format Support:** Ingest `.pdf`, `.docx`, and `.txt` files seamlessly.
* **💬 Integrated Chat:** Connect any OpenAI-compatible API to generate answers based on your uploaded knowledge base.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- (Optional) An API Key for an LLM provider (OpenRouter, OpenAI, etc.)

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/edurag-glassbox.git](https://github.com/yourusername/edurag-glassbox.git)
   cd edurag-glassbox
   ```
2. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn numpy scikit-learn httpx pypdf python-docx
   ```
3. **Launch the application:**
   ```bash
    python main.py
   ```
4. **Access the UI:**
Open your browser and navigate to http://localhost:8000



## 🚀 Tech Stack

Layer

Technology

Backend Framework

FastAPI (Python)

NLP & Vectorization

Scikit-Learn (TF-IDF Vectorizer)

Math & Logic

NumPy & Cosine Similarity

Visualization

Plotly.js (3D Scatter Plots)

UI Styling

Tailwind CSS & FontAwesome

File Parsing

pypdf & python-docx

## ⚙️ Architecture: The 5-Step Pipeline

Ingest: Raw documents are extracted and cleaned.

Chunk: Text is segmented into overlapping windows to preserve semantic context.

Embed: Chunks are vectorized into a high-dimensional feature space using TF-IDF.

Retrieve: The system calculates the cosine angle between the user query vector and the document vectors.

Generate: Top-K relevant chunks are synthesized into a grounded response via an LLM.



## 📄 License

Distributed under the MIT License. See LICENSE for more information.

Developed for the AI Education community.
