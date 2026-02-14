import streamlit as st
import os
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv("GOOGLE_API_KEY.env")
api_key = os.getenv("GOOGLE_API_KEY")

# 2. Setup Page
st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
st.header("🤖 Chat with your PDF")

mode = st.sidebar.radio("Bot Personality:", ["Strict RAG (Business)", "Helpful Assistant (Open)"])

if not api_key:
    st.error("❌ API Key is missing!")
    st.stop()

# --- MODERN IMPORTS (Avoids .chains) ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 3. THE BRAIN ---
@st.cache_resource
def load_rag_pipeline(user_mode):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    try:
        vector_store = FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ Index Error: {e}")
        st.stop()
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)

    # Define Prompts
    if user_mode == "Strict RAG (Business)":
        prompt_text = "You are a strict analyst. Use ONLY the context: {context}\nQuestion: {question}"
    else:
        prompt_text = "Use context first. If not there, use general knowledge but say 'Not in PDF, but...':\nContext: {context}\nQuestion: {question}"

    prompt = ChatPromptTemplate.from_template(prompt_text)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # --- THE MODERN CHAIN (LCEL) ---
    # This replaces RetrievalQA and avoids the 'langchain.chains' error
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# Initialize
with st.spinner("Loading..."):
    rag_chain, retriever = load_rag_pipeline(mode)

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Generate Answer
        response_text = rag_chain.invoke(user_query)
        st.markdown(response_text)
        
        # Show Sources manually
        # Show Sources manually
        with st.expander("📚 View Sources"):
            # OLD: source_docs = retriever.get_relevant_documents(user_query)
            # NEW: We use .invoke() just like we did with the chain
            try:
                source_docs = retriever.invoke(user_query)
                for doc in source_docs:
                    st.write(f"**Source Chunk:** {doc.page_content[:150]}...")
            except Exception as e:
                st.write("Could not load sources.")