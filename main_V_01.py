import os
import re
import requests
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_conversation_via_email(email, chat_history):
    msg = MIMEMultipart()
    msg['From'] = "beyondrelativity8@gmail.com"
    msg['To'] = email
    msg['Subject'] = "Your Dreamspire Chat Summary"

    body = "\n".join(
        f"{'You' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in chat_history
    )

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login("beyondrelativity8@gmail.com", "nmvh bcxe bekg izzi")
            server.sendmail(msg['From'], msg['To'], msg.as_string())
        return True
    except Exception as e:
        print(f"Email send error: {e}")
        return False


def save_to_excel(memory_dict, chat_history):
    summary_text = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in chat_history
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    row = {
        "Timestamp": timestamp,
        "Name": memory_dict.get("name", ""),
        "Email": memory_dict.get("email", ""),
        "Phone": memory_dict.get("phone", ""),
        "Conversation Summary": summary_text
    }

    file_path = "chat_logs.xlsx"

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    df.to_excel(file_path, index=False)

# Load and split documents
def load_docs():
    all_docs = []

    for folder, tag in [("park_data", None), ("Conversation Data", "sample_conversation"), ("Guidelines","Guidelines")]:
        for filename in os.listdir(folder):
            if filename.endswith(".pdf"):
                path = os.path.join(folder, filename)
                loader = PyPDFLoader(path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    if tag:
                        doc.metadata["type"] = tag
                all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_documents(all_docs)

# Build retrievers
def build_retrievers(splits):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    vector_retriever = vectorstore.as_retriever()
    keyword_retriever = BM25Retriever.from_documents(splits)

    return vectorstore.as_retriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.7, 0.3]
    )

def extract_info(user_input):
    updates = {}
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", user_input)
    if email_match:
        updates["email"] = email_match.group(0)
    phone_match = re.search(r"(\+?\d[\d\s\-]{7,}\d)", user_input)
    if phone_match:
        updates["phone"] = phone_match.group(0).strip()
    name_match = re.search(r"(?:my name is|I'm|I am)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", user_input)
    if name_match:
        updates["name"] = name_match.group(1)
    return updates

# Prompt template (updated to include memory and history)
prompt = PromptTemplate(
    input_variables=["context", "question", "memory", "history"],
    template="""
You are a friendly, casual, and human-sounding online sales & support assistant for Dreamspire Amusement Park — a large, magical, hyperrealistic theme park filled with fantasy worlds, IP-based zones, and premium guest experiences.

Speak like a real person: casual, helpful, fun, and sometimes with small typos or grammar mistakes to sound more natural. Your tone should always feel human, supportive, and enthusiastic.

You must follow the official Dreamspire Chatbot Behavior Guidelines stored in the "Guidelines" folder (PDFs). These include:

- How to speak and what tone to use
- How to respond based on visitor type (e.g., family, large group, influencer, solo traveler)
- When and how to offer follow-up questions
- How to collect email at the end of a conversation
- What to offer in special situations (like group dining, custom events, child care areas, etc.)

Use those documents as the source of truth for how you communicate and behave during all interactions.

Always:
- Give helpful, conversational answers
- Adjust your response based on the type of visitor
- Ask if they had like to receive this conversation via email
- Offer extra relevant information

Never:
- Sound robotic or overly perfect
- Overpromise unavailable services
- Skip questions or ignore context

Memory:  
{memory}

Previous Chat:  
{history}

----
[Park Info]  
{context}
----

Customer: {question}  
You:

"""
)

# Memory extraction
def extract_info(user_input):
    updates = {}
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", user_input)
    if email_match:
        updates["email"] = email_match.group(0)
    phone_match = re.search(r"(\+?\d[\d\s\-]{7,}\d)", user_input)
    if phone_match:
        updates["phone"] = phone_match.group(0).strip()
    return updates

def format_memory(memory_dict):
    return "\n".join(f"{k.capitalize()}: {v}" for k, v in memory_dict.items())

# Ask Ollama
def ask_ollama(context, memory, history, question):
    filled_prompt = prompt.format(context=context, memory=memory, history=history, question=question)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": filled_prompt, "stream": False}
    )
    return response.json().get("response", "No response from model.")

# Load docs and build retriever
splits = load_docs()
retriever = build_retrievers(splits)

# Streamlit App
st.set_page_config(page_title="Dreamspire Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center;'>Dreamspire Assistant </h1>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_memory" not in st.session_state:
    st.session_state.user_memory = {}

with st.form("chat_input", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Ask me anything about Dreamspire!", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    updates = extract_info(user_input)
    st.session_state.user_memory.update(updates)

    memory = format_memory(st.session_state.user_memory)
    history = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in st.session_state.chat_history[-6:]
    ])

    relevant_docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    with st.spinner("Thinking..."):
        answer = ask_ollama(context, memory, history, user_input)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "bot", "content": answer})
    save_to_excel(st.session_state.user_memory, st.session_state.chat_history)

# Chat Display
for msg in st.session_state.chat_history:
    style = "text-align: right;" if msg["role"] == "user" else "text-align: left;"
    st.markdown(f"""
    <div style="{style} background-color: #000000; padding: 10px; border-radius: 10px; margin: 5px;">
        <strong>{'You' if msg['role'] == 'user' else 'Assistant'}:</strong> {msg['content']}
    </div>
    """, unsafe_allow_html=True)

if st.session_state.user_memory.get("email"):
    if st.button(" Send Chat to My Email"):
        sent = send_conversation_via_email(
            st.session_state.user_memory["email"],
            st.session_state.chat_history
        )
        if sent:
            st.success("Chat sent to your email! ")
        else:
            st.error("Failed to send email. Please try again later.")
