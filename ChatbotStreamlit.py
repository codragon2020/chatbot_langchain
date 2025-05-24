from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from dotenv import load_dotenv
import streamlit as st
import os


load = load_dotenv('.\..\.env')

# llm = ChatOllama(
#     base_url="http://localhost:11434",
#     model = "mistral:latest",
#     temperature=0.5,
#     max_tokens = 250
# )

openaiKey = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4.1", 
                 openai_api_key=openaiKey, 
                 temperature=0.5, 
                 max_tokens=1000)


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id=session_id, connection="sqlite:///chat_history.db")

user_id = "Jason"

with st.sidebar:
    user_id = st.text_input("Enter your name", user_id)
    role = st.radio("How detailed should your answer be?", ["Beginner", "Expert", "PHD"], index=0)
    if st.button("Start new chat"):
        st.session_state.chat_history = []
        get_session_history(user_id).clear()

st.markdown(
    """
    <div style='display: flex; height: 70vh; justify-conten: center; align-items: center;'>
        <h1 style='text-align: center; font-size: 50px; color: #4B0082;'>AI Ready RVA Chatbot</h1> 
    """
    , unsafe_allow_html=True
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ('system', f"You are an {role} AI assistant that answers questions about AI Ready RVA."),
    ('human', "{prompt}"),
])

chain = template | llm | StrOutputParser()

def invoke_history(chain, session_id, prompt):
    history = RunnableWithMessageHistory(
        chain, 
        get_session_history, 
        input_messages_key="prompt",
        history_messages_key="history"
    )

    for response in history.stream({"prompt": prompt}, config={"configurable": {"session_id": session_id}}):
        yield response

prompt = st.chat_input("Ask me anything about AI Ready RVA")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
            stream_response = st.write_stream(invoke_history(chain, user_id, prompt))
    
    st.session_state.chat_history.append({"role": "assistant", "content": stream_response})

# Clear the chat history after the conversation
# get_session_history(user_id).clear()