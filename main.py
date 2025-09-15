# main.py
import streamlit as st
import os
import re

# --------------------------
# Groq + LangChain imports
# --------------------------
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Updated imports from langchain_community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# --------------------------
# Streamlit page setup
# --------------------------
st.set_page_config(page_title="Sudip's ChatBot", page_icon="🤖")
st.title("ChatBot by Sudip Paneru")

# --------------------------
# Initialize session state for chat messages
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# --------------------------
# PDF Upload UI
# --------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# --------------------------
# Vectorstore creation
# --------------------------
@st.cache_resource
def get_vectorstore(uploaded_files):
    """
    Convert uploaded PDFs into a vectorstore using embeddings
    """
    loaders = []

    for uploaded_file in uploaded_files:
        # Save PDF temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Load PDF into PyPDFLoader
        loader = PyPDFLoader(temp_path)
        loaders.append(loader)

    if not loaders:
        return None

    # Create vectorstore using embeddings
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)

    return index.vectorstore

# --------------------------
# Helper function to clean model output
# --------------------------
def clean_response(text):
    """
    Removes <think>...</think> blocks from Groq output
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned

# --------------------------
# User prompt input
# --------------------------
prompt = st.chat_input("Type your question here...")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --------------------------
    # System prompt for Groq
    # --------------------------
    groq_sys_prompt = ChatPromptTemplate.from_template("""
    You are a professional virtual assistant. Only provide the final answer.
    Do NOT include any internal thoughts or reasoning.
    Query: {user_prompt}
    """)

    # --------------------------
    # Initialize ChatGroq
    # --------------------------
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not set in environment.")
        st.stop()  # Stop execution if API key missing

    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model="qwen/qwen3-32b"
    )

    try:
        # --------------------------
        # Create vectorstore if PDFs uploaded
        # --------------------------
        vectorstore = None
        if uploaded_files:
            vectorstore = get_vectorstore(uploaded_files)
            if vectorstore is None:
                st.error("Failed to process uploaded PDFs")

        # --------------------------
        # If PDFs uploaded, use RetrievalQA
        # --------------------------
        if vectorstore:
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            result = chain({"query": prompt})
            raw_response = result["result"]
            response = clean_response(raw_response)
        else:
            # --------------------------
            # Otherwise, just use system prompt
            # --------------------------
            chain = groq_sys_prompt | groq_chat | StrOutputParser()
            raw_response = chain.invoke({"user_prompt": prompt})
            response = clean_response(raw_response)

        # --------------------------
        # Display assistant response
        # --------------------------
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {str(e)}")
