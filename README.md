# Universal PDF Bot: RAG-Powered Document Analysis

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot capable of analyzing and discussing PDF documents in natural language. Built to bridge the gap between static documents and dynamic queries, the application allows users to upload contracts, research papers, or slides and receive instant, citation-backed answers.

The system leverages **Google's Gemini** models for reasoning and **FAISS** for vector retrieval, featuring a unique "Dual-Mode" system that adapts the AI's behavior for either strict legal analysis or helpful general summarization.

## Technical Methodology

### RAG Pipeline Architecture

* **Generative Model:** Google Gemini (via `langchain-google-genai`)
* **Vector Database:** FAISS (Facebook AI Similarity Search) for efficient similarity search
* **Embeddings:** Google Generative AI Embeddings (`models/gemini-embedding-001`)
* **Orchestration:** LangChain for managing the retrieval and generation chains

### Key Technical Features

1.  **Adaptive Rate Limiting Strategy:**
    To handle Google API constraints efficiently, the system automatically classifies documents by size and adjusts processing speed:
    * **Fast Mode (<50 chunks):** High concurrency for rapid indexing.
    * **Balanced Mode (50-150 chunks):** Moderate throttling.
    * **Safe Mode (>150 chunks):** Aggressive delays to prevent `429 Resource Exhausted` errors.

2.  **Dual-Mode Inference:**
    * **Strict Mode (Temperature 0.2):** Designed for contracts/legal docs. Forces the model to quote exact wording and avoids hallucination.
    * **Helpful Mode (Temperature 0.6):** Designed for slides/books. Allows for synthesis, summarization, and natural conversation.

3.  **Smart Caching & Hashing:**
    * Implements MD5 file hashing to detect duplicate uploads.
    * Uses `Streamlit.cache_resource` to prevent re-processing the same PDF during a session.

## Repository Structure

```bash
.
├── app_v1.py                 # Main application logic (Streamlit frontend + RAG backend)
├── requirements.txt          # Python dependencies
├── .gitignore                # Security configuration (excludes API keys)
└── README.md                 # Project documentation
```

## Installation and Requirements
**Python Version**
* Python 3.9+ recommended

**Required Libraries**
Core dependencies include streamlit, langchain, google-generativeai, and faiss-cpu.

```bash
pip install -r requirements.txt
```

## API Configuration
This project requires a Google Gemini API key.
* Create a .env file in the root directory.
* Add your key:
  GOOGLE_API_KEY=your_api_key_here

## Usage
**Option 1: Run Locally**
* Clone the repository:

```bash
git clone [https://github.com/burhanuddink/Universal-pdf-bot.git](https://github.com/burhanuddink/Universal-pdf-bot.git)
cd Universal-pdf-bot
```
* Install dependencies:

```bash
pip install -r requirements.txt
```
* Run the application:

```bash
streamlit run app_v1.py
```
* Access the App: Open your browser to http://localhost:8501.

**Option 2: Live Demo (Streamlit Cloud)**
This application is deployed on Streamlit Community Cloud.
* Click here to view the Live Demo (https://universal-pdf-bot.streamlit.app/)
* Note: You may need to enter your own API key in the sidebar if the public demo key is exhausted.

## Experimental Results
**Performance Metrics**
* Retrieval Accuracy: Utilizes k=10 retrieval with a similarity score threshold to ensure high context relevance.

* Processing Speed:
  * Small Docs (<10 pages): ~2-5 seconds.
  * Large Docs (>50 pages): ~45-60 seconds (throttled for stability).

**User Experience**
* Source Transparency: Every answer includes an expandable "View Sources" section, citing the exact page number and text snippet used to generate the response.
* Error Handling: Robust handling for API timeouts, file corruption, and empty OCR scans.

## Future Improvements
* Persistent Storage: Transition from in-memory FAISS to persistent vector storage (ChromaDB or Pinecone) to allow cross-session memory.
* Multi-File Support: Enable querying across multiple PDFs simultaneously.
* OCR Integration: Add Tesseract support to handle scanned/image-based PDFs that currently fail text validation.

## Acknowledgements
* LangChain Framework for the RAG pipeline construction.
* Google GenAI for providing the Gemini inference and embedding models.
* Streamlit for the rapid UI development framework.
