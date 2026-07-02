import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------- Environment variables ----------
os.environ["USER_AGENT"] = "MyRAGApp/1.0"

# ---------- Page config ----------
st.set_page_config(
    page_title="RAG Web Search Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    url = st.text_input(
        "📄 Source URL",
        value="https://www.ppsc.gop.pk/",
        help="Website to index and query."
    )
    
    st.markdown("---")
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value="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",  # Replace with your actual key
        help="Your Groq API key."
    )
    
    model_name = st.selectbox(
        "🧠 LLM Model",
        ["openai/gpt-oss-120b", "mixtral-8x7b-32768", "llama2-70b-4096"],
        index=0
    )
    
    st.markdown("---")
    k = st.slider("📊 Retrieved chunks (k)", min_value=1, max_value=10, value=5)
    chunk_size = st.slider("📏 Chunk size", min_value=200, max_value=1000, value=500, step=50)
    
    st.markdown("---")
    st.caption("Built with LangChain, FAISS, Groq & Streamlit")

# ---------- Cached resource loading ----------
@st.cache_resource
def load_resources(url, chunk_size, k):
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever

@st.cache_resource
def load_llm(api_key, model_name):
    return ChatGroq(model=model_name, api_key=api_key)

def build_rag_chain(retriever, llm):
    system_prompt = (
        "You are a helpful assistant. Provide answers based on the provided context. "
        "If the information is not in the context, use your intelligence to answer the questions."
        "\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# ---------- Main UI ----------
st.title("🔍 RAG Web Search Assistant")
st.markdown("Ask questions based on the content of the website you provided.")

# Load resources
with st.spinner("🔄 Loading and indexing the website... This may take a moment."):
    try:
        retriever = load_resources(url, chunk_size, k)
        llm = load_llm(groq_api_key, model_name)
        rag_chain = build_rag_chain(retriever, llm)
        st.success("✅ Ready! Ask your question below.")
    except Exception as e:
        st.error(f"❌ Error loading resources: {e}")
        st.stop()

# ---------- Conversation history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "context" in msg:
            with st.expander("📖 View retrieved context"):
                st.write(msg["context"])

# ---------- Chat input ----------
query = st.chat_input("Type your question here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
                context_docs = response.get("context", [])
                context_text = "\n\n".join([doc.page_content for doc in context_docs]) if context_docs else "No context retrieved."

                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "context": context_text
                })
                with st.expander("📖 View retrieved context"):
                    st.write(context_text)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")