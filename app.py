import streamlit as st
import os
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
os.environ["USER_AGENT"] = "MyGitLabChatBot/1.0"

# Securely pull the key from Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("API Key not found! Please add it to Streamlit Secrets.")
    st.stop()

GITLAB_URLS =[
    "https://about.gitlab.com/handbook/values/",
    "https://about.gitlab.com/handbook/total-rewards/benefits/general-working/",
    "https://about.gitlab.com/direction/ai-powered/"
]

st.set_page_config(page_title="GitLab Employee Assistant", page_icon="🦊", layout="wide")

with st.sidebar:
    st.image("https://about.gitlab.com/images/press/logo/png/gitlab-icon-rgb.png", width=50)
    st.title("GitLab Assistant")
    st.markdown("Built using Streamlit & Gemini")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages =[]
        st.session_state.chat_history =[]
        st.rerun()

st.title("GitLab Handbook Assistant")

@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    with st.spinner("Initializing Vector Database..."):
        loader = WebBaseLoader(GITLAB_URLS)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore

vectorstore = get_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1) 

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question. Do NOT answer it, just reformulate it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an official GitLab internal assistant. 
    GUARDRAIL: You must ONLY answer questions based on the provided context. 
    If the user asks a question entirely unrelated to GitLab or the context, politely reply: 'I am a GitLab assistant and can only answer questions related to GitLab's Handbook and Direction.'
    
    Context: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

if "messages" not in st.session_state:
    st.session_state.messages =[{"role": "assistant", "content": "Welcome! I can answer questions about GitLab's Values, Remote Work Benefits, and AI Strategy. How can I help?"}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history =[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about GitLab's remote work or values..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching Handbook..."):
            try:
                result = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
                answer = result["answer"]
                
                sources = set([doc.metadata.get('source', 'Unknown') for doc in result['context']])
                source_text = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
                full_response = answer + source_text
                
                st.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.chat_history.extend([HumanMessage(content=user_input), AIMessage(content=answer)])
            except Exception as e:
                st.error("I'm sorry, I encountered an error connecting to the AI. Please try again.")
