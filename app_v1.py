import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import time
from datetime import datetime
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Page Setup
st.set_page_config(
    page_title="Universal PDF Bot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header("🤖 Universal PDF Chatbot")

load_dotenv("API_KEY.env")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key and hasattr(st, 'secrets'):
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        pass

if not api_key:
    st.error("❌ API Key is missing!")
    st.info("""
    **Setup Instructions:**
    - **Local development:** Create a `.env` file with `GOOGLE_API_KEY=your_key`
    - **Streamlit Cloud:** Add `GOOGLE_API_KEY` in your app secrets
    """)
    st.stop()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_file_hash(file_bytes):
    """Generate hash for cache invalidation"""
    return hashlib.md5(file_bytes).hexdigest()[:16]

# PDF processing with adaptive rate limiting and error handling
@st.cache_resource(show_spinner=False)
def process_pdf(file_name, _file_bytes):
    """Process PDF with adaptive rate limiting"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        if not pages:
            raise ValueError("Could not load PDF. It might be corrupted or password-protected.")
        
        total_text = "".join([page.page_content for page in pages])
        if len(total_text.strip()) < 100:
            raise ValueError("PDF appears to be scanned or empty. Please use an OCR tool first.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)
        num_chunks = len(chunks)
        
        if num_chunks <= 50:
            batch_size, delay_seconds, mode = 25, 0.3, "Fast"
        elif num_chunks <= 150:
            batch_size, delay_seconds, mode = 10, 1.5, "Balanced"
        else:
            batch_size, delay_seconds, mode = 5, 4, "Safe"
            est_time = (num_chunks * 0.7) / 60
            st.warning(f"📄 Large PDF: ~{est_time:.1f} minutes to process")
        
        st.info(f"📄 {mode} mode: {len(pages)} pages → {num_chunks} chunks")
        logger.info(f"Processing {file_name}: {len(pages)} pages, {num_chunks} chunks, {mode} mode")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        
        progress_bar = st.progress(0)
        vector_store = None
        start_time = datetime.now()
        retry_count = 0
        
        i = 0
        while i < num_chunks:
            batch = chunks[i:i+batch_size]
            
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                
                i += batch_size
                retry_count = 0
                
                progress = min(i / num_chunks, 1.0)
                elapsed = (datetime.now() - start_time).seconds
                progress_bar.progress(
                    progress, 
                    text=f"{min(i, num_chunks)}/{num_chunks} chunks ({elapsed}s)"
                )
                
                if i < num_chunks:
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    retry_count += 1
                    wait_time = 8 if retry_count == 1 else 12
                    st.warning(f"⏳ Rate limit - waiting {wait_time}s... (attempt {retry_count})")
                    logger.warning(f"Rate limit hit at chunk {i}, retry {retry_count}")
                    time.sleep(wait_time)
                    
                    if retry_count > 1:
                        batch_size, delay_seconds = 3, 5
                    continue
                else:
                    logger.error(f"Error processing batch at chunk {i}: {error_str}")
                    raise e
        
        progress_bar.empty()
        total_time = (datetime.now() - start_time).seconds
        st.success(f"✅ Processed in {total_time}s")
        logger.info(f"Successfully processed {file_name} in {total_time}s")
        
        return vector_store
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# RAG CHAIN
def get_rag_chain(vector_store, user_mode):
    """Build RAG chain with mode-specific configuration"""
    
    temp = 0.2 if user_mode == "Strict (Contracts/Legal)" else 0.6
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key, 
        temperature=temp
    )
    
    if user_mode == "Strict (Contracts/Legal)":
        temp = 0.2
        prompt_text = """You are a precise analyst answering based ONLY on the context below.

        Quoting rules:
        - Quote exact wording for: legal terms, contract clauses, technical definitions, award names
        - DO NOT quote: years, numbers, biographical facts, general descriptions
        - Synthesize most information naturally without quotes
        
        Example - BAD: He was born in "1962" and published "300 papers"
        Example - GOOD: He was born in 1962 and published over 300 papers
        
        Context: {context}
        Question: {question}
        
        Answer clearly with minimal, purposeful quoting:"""
        
    else:
        temp = 0.6
        prompt_text = """You are a helpful assistant explaining information clearly.

        Write in natural, flowing prose as if talking to a colleague:
        - Synthesize and paraphrase - don't quote mundane facts
        - Only quote specific terms, unique phrases, or important exact wordings
        - Never quote years, numbers, or common biographical information
        - Make it readable and conversational
        
        Bad example: He received a "B.S." from Brown in "1975"
        Good example: He received a B.S. from Brown in 1975
        
        Context: {context}
        Question: {question}
        
        Provide a natural, well-written answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# Sidebar for settings and file upload
with st.sidebar:
    st.title("⚙️ Settings")
    
    mode = st.radio(
        "Bot Mode:", 
        ["Strict (Contracts/Legal)", "Helpful (Slides/General)"],
        help="Strict: Only uses PDF content. Helpful: Can use general knowledge."
    )
    
    st.divider()
    
    uploaded_file = st.file_uploader(
        "📄 Upload PDF", 
        type="pdf",
        help="Maximum 50MB",
        max_upload_size=50
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    st.divider()
    st.caption("💡 **Tips:**")
    st.caption("• Use Strict for legal docs")
    st.caption("• Use Helpful for slides/books")
    st.caption("• Clear cache if switching PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

# MAIN APP
if uploaded_file:
    max_size_mb = 50
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        st.error(f"❌ File too large ({file_size_mb:.1f}MB). Maximum: {max_size_mb}MB")
        st.stop()
    
    current_hash = get_file_hash(file_bytes)
    
    if st.session_state.current_file_hash != current_hash:
        st.session_state.current_file = uploaded_file.name
        st.session_state.current_file_hash = current_hash
        st.session_state.messages = []
        st.info(f"📄 New file: {uploaded_file.name} ({file_size_mb:.1f}MB)")

    try:
        with st.spinner("Processing PDF..."):
            vector_store = process_pdf(uploaded_file.name, file_bytes)
        
        rag_chain, retriever = get_rag_chain(vector_store, mode)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_query := st.chat_input("Ask about your document..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                try:
                    response_text = rag_chain.invoke(user_query)
                    st.markdown(response_text)

                    with st.expander("🔍 View Sources"):
                        sources = retriever.invoke(user_query)
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', '?')})**")
                            st.text_area(
                                f"Excerpt {i+1}", 
                                doc.page_content[:400] + "...",
                                height=100,
                                disabled=True,
                                key=f"src_{current_hash}_{i}_{len(st.session_state.messages)}"
                            )
                    
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    logger.info(f"Query processed: {user_query[:50]}...")
                    
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error: {error_msg}")
                    logger.error(f"Query failed: {error_msg}", exc_info=True)

                    if "429" in error_msg:
                        st.warning("⏳ Rate limit reached. Please wait a minute.")
                    elif "context_length" in error_msg.lower():
                        st.info("💡 Try asking a more specific question.")
    
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Failed to process PDF")
        logger.error(f"PDF processing failed: {error_msg}", exc_info=True)

        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            st.warning("⏳ API rate limit reached. Please wait a minute and reload the page.")
        elif "NOT_FOUND" in error_msg:
            st.warning("⚠️ API configuration issue. Please verify your API key and model settings.")
        elif "password" in error_msg.lower():
            st.info("💡 This PDF is password-protected. Please unlock it first.")
        else:
            st.info("""
            **Troubleshooting:**
            - Ensure PDF contains extractable text (not scanned)
            - Check if PDF is password-protected
            - Try a smaller PDF first
            - Clear cache and try again
            """)

else:
    st.info("""
    👋 **Welcome! Upload a PDF to start chatting.**
    
    **Works with:**
    - 📄 Legal contracts & agreements
    - 📊 Presentation slides
    - 📚 Reports & textbooks
    - 📋 Research papers
    
    **Features:**
    - Smart adaptive processing
    - Source citations
    - Two modes: Strict & Helpful
    """)

    with st.expander("💡 Example Questions"):
        st.markdown("""
        **For Contracts:**
        - "What is the termination clause?"
        - "What are the payment terms?"
        
        **For Slides:**
        - "Summarize the key findings"
        - "What are the main recommendations?"
        
        **For Papers:**
        - "What methodology was used?"
        - "What are the conclusions?"
        """)