import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Template for question-answering
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Directory to store uploaded PDFs
pdfs_directory = 'chat-with-pdf/pdfs/'

# Ensure the directory exists
if not os.path.exists(pdfs_directory):
    os.makedirs(pdfs_directory)

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)

# Initialize the model
model = OllamaLLM(model="deepseek-r1:1.5b")

# Function to save uploaded PDF
def upload_pdf(file):
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

# Function to load PDF content
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# Function to split text into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# Function to index documents
def index_docs(documents):
    vector_store.add_documents(documents)

# Function to retrieve documents based on query
def retrieve_docs(query):
    return vector_store.similarity_search(query)

# Function to answer questions
def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Streamlit file uploader
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    try:
        # Save and process the uploaded file
        file_path = upload_pdf(uploaded_file)
        documents = load_pdf(file_path)
        chunked_documents = split_text(documents)
        index_docs(chunked_documents)

        # User input for question
        question = st.chat_input()

        if question:
            st.chat_message("user").write(question)
            related_documents = retrieve_docs(question)
            answer = answer_question(question, related_documents)
            st.chat_message("assistant").write(answer)

    except Exception as e:
        st.error(f"An error occurred: {e}")
