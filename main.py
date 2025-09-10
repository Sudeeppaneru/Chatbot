#phase 1 imports
import streamlit as st

#phase 2 imports
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#phase 3 imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

st.title("The ChatBot by Sudip Paneru")

#   Setup a session state variabe to hold all the old messages
if 'messages'   not in st.session_state:
        st.session_state.messages    =   []

#   Display old messages
for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
        pdf_name = []
        loaders = {PyPDFLoader(pdf_name)}
        # Create chunks, aka vectors (ChromaDb)
        index = VectorstoreIndexCreator(
                embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
                text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)   
        ).from_loaders(loaders)
        return index.vectorstore

#   Prompt From User
prompt  =   st.chat_input("Prompt here")

#   If prompt is available
if prompt:
        #   Pass the prompt to llm and Save the prompt and response
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})

        groq_sys_prompt = ChatPromptTemplate.from_template(""" You are a an virtual assastent that helps in the company :{user_prompt}
                                                           """)
        model = "llama3-8b-8192"
        groq_chat   = ChatGroq(
                groq_api_key = os.environ.get("GROQ_API_KEY")

        )

        try:
             vectorstore = get_vectorstore()
             if vectorstore is None:
                st.error("Failed to load the document")
             
             chain = RetrievalQA.from_chain_type(
                    llm = groq_chat,
                    chain_type = 'stuff',
                    retriever = vectorstore.as_retriver(search_kwargs = ({"k":3})),
                    return_source_document = True
             )
             result = chain({"query":prompt})
             response = result["result"]

        
             chain = groq_sys_prompt | groq_chat | StrOutputParser()
             response = chain.invoke({"user_prompt":prompt})

                #response    =   "I am your assistant"
             st.chat_message('assistant').markdown(response)
             st.session_state.messages.append({'role':'assistant','content':response})

        except Exception as e:
                st.error(f"Error: {str(e)}")