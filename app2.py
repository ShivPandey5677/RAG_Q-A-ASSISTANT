import os
import math
import requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import tiktoken

# Load environment variables
load_dotenv()
# keep langchain and openai api key in .env file

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    return dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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
            return "⚠️ Could not evaluate the expression."
    except Exception as e:
        return f"Error: {e}"
def load_documents(folder="./docs"):
    documents = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(folder, fname), encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(documents)

def create_vector_store(chunks):
    return Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
def main():
    st.set_page_config(page_title="RAG Assistant", layout="wide")
    st.title("RAG-Powered Multi-Agent Q&A Assistant")

    if "vectorstore" not in st.session_state:
        raw_docs = load_documents()
        if not raw_docs:
            st.error("No documents found in './docs'. Please add .txt files.")
            return
        chunks = chunk_documents(raw_docs)
        st.session_state.vectorstore = create_vector_store(chunks)
        st.session_state.qa_chain = build_qa_chain(st.session_state.vectorstore)

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
            retriever = st.session_state.vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(user_query)
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = st.session_state.qa_chain.invoke(user_query)

        # Display results
        st.subheader("Decision")
        st.write(decision)

        if context:
            st.subheader("Retrieved Context")
            st.write(context)

        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
