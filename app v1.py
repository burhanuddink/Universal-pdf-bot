import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- 1. SETUP PAGE & ENV ---
st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
st.header("🤖 Chat with ANY PDF")

# Load API Key
load_dotenv("GOOGLE_API_KEY.env")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ API Key is missing! Check GOOGLE_API_KEY.env")
    st.stop()

# --- 2. IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 3. THE INGESTION ENGINE (New!) ---
# This function handles the file upload -> processing -> vector store creation
@st.cache_resource
def process_pdf(uploaded_file):
    # A. Save uploaded file to a temporary file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # B. Load and Split the PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    
    # C. Create Vector Store (This calls Google Embeddings API)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Cleanup temp file
    os.remove(tmp_path)
    
    return vector_store

# --- 4. THE RAG PIPELINE (Updated) ---
# Now accepts 'vector_store' as an argument instead of loading from disk
def get_rag_chain(vector_store, user_mode):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
    
    # Define Prompts
    if user_mode == "Strict RAG (Business)":
        prompt_text = """You are a professional analyst. 
        Answer the question based ONLY on the context provided below. 
        Do not answer from memory.
        If the answer is not in the context, say: "I cannot find this answer in the document."
        
        Context: {context}
        Question: {question}
        """
    else:
        # UPDATED "HELPFUL" PROMPT
        prompt_text = """You are a helpful research assistant. 
        Use the context provided below to answer the question.
        
        Instructions:
        1. Read the context and synthesize the information into a clear, natural answer.
        2. Do not just list bullet points; try to explain the concepts.
        3. If the answer is not in the context, use your general knowledge, but start with: "The document doesn't mention this explicitly, but generally..."
        
        Context: {context}
        Question: {question}
        """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    # INCREASED k TO 10 (Better for slide decks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Build Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# --- 5. SIDEBAR & FILE UPLOAD ---
with st.sidebar:
    st.title("Settings")
    mode = st.radio("Bot Personality:", ["Strict RAG (Business)", "Helpful Assistant (Open)"])
    st.divider()
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- 6. MAIN APP LOGIC ---
if uploaded_file:
    # Step 1: Process the File
    with st.spinner("Processing PDF... (This creates embeddings)"):
        # We process the PDF and cache the result so it doesn't run on every question
        vector_store = process_pdf(uploaded_file)
        st.success("PDF Processed!")

    # Step 2: Build the Chain
    rag_chain, retriever = get_rag_chain(vector_store, mode)

    # Step 3: Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                # Get Answer
                response_text = rag_chain.invoke(user_query)
                st.markdown(response_text)
                
                # Get Sources
                with st.expander("📚 View Sources"):
                    sources = retriever.invoke(user_query)
                    for doc in sources:
                        st.write(f"**Page {doc.metadata.get('page', '?')}**")
                        st.caption(doc.page_content[:150] + "...")
                        
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                st.error(f"Error: {e}")

else:
    # Welcome Screen
    st.info("👋 Please upload a PDF file in the sidebar to start chatting!")