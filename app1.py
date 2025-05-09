
import os
import math
import requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def define_term(query):
    term = query.lower().replace("define", "").strip()
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}"
    try:
        response = requests.get(url)
        data = response.json()
        if isinstance(data, list):
            definition = data[0]["meanings"][0]["definitions"][0]["definition"]
            return f"**{term}**: {definition}"
        else:
            return f"No definition found for '{term}'."
    except Exception as e:
        return f"Error fetching definition: {e}"

def calculate_expression(query):
    expression = query.lower().replace("calculate", "").strip()
    url = f"https://api.mathjs.org/v4/?expr={requests.utils.quote(expression)}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return f"The result of `{expression}` is {response.text}"
        else:
            return "Could not evaluate the expression."
    except Exception as e:
        return f"Error: {e}"

def load_documents(folder="./docs"):
    documents = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            loader = TextLoader(os.path.join(folder, fname), encoding='utf-8')
            documents.extend(loader.load())
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(documents)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

def mock_llm_response(context, question):
    return f"(This is a mock response based on context.)\n\n{context[:300]}..."
def main():
    st.set_page_config(page_title="RAG Assistant", layout="wide")
    st.title("RAG Q&A Assistant (with Public Tools)")

    if "vectorstore" not in st.session_state:
        docs = load_documents()
        if not docs:
            st.error("No documents found in './docs'. Please add .txt files.")
            return
        chunks = chunk_documents(docs)
        st.session_state.vectorstore = create_vector_store(chunks)
        st.session_state.logs = []

    user_query = st.text_input("Ask a question:")

    if user_query:
        query_lower = user_query.lower()
        decision = "RAG Pipeline"
        context = None

        if "define" in query_lower:
            decision = "Dictionary Tool"
            answer = define_term(user_query)
        elif "calculate" in query_lower:
            decision = "Calculator Tool"
            answer = calculate_expression(user_query)
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_query)
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = mock_llm_response(context, user_query)

        st.session_state.logs.append({"query": user_query, "decision": decision})

        st.subheader("Decision")
        st.write(decision)

        if context:
            st.subheader("Retrieved Context")
            st.write(context)

        st.subheader(" Answer")
        st.write(answer)

        with st.expander("Interaction Log"):
            for log in st.session_state.logs:
                st.markdown(f"- **Query:** {log['query']} | **Decision:** {log['decision']}")

if __name__ == "__main__":
    main()
