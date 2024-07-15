import streamlit as st
import requests

st.set_page_config(page_title="Chat with AI", page_icon=":robot_face:", layout="centered")

# Custom CSS for chat interface
st.markdown("""
    <style>
    .chat-box {
        max-height: 70vh;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #f7f7f7;
    }
    .message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
    }
    .user {
        background-color: #e1ffc7;
        text-align: right;
    }
    .bot {
        background-color: #dcdcdc;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
    }
    .input-container textarea {
        width: 80%;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ddd;
        resize: none;
    }
    .input-container button {
        width: 18%;
        padding: 10px;
        border: none;
        border-radius: 8px;
        background-color: #007bff;
        color: white;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

st.title("AI Chat Interface")

# Initialize session state for history if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = []

# Display history of the conversation
st.markdown('<div class="chat-box">', unsafe_allow_html=True)
for item in st.session_state.history:
    st.markdown(f'<div class="message user"><b>You:</b> {item["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="message bot"><b>AI:</b> {item["answer"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Text input and send button at the bottom
with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_area("Your message:", key='input', height=70)
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    with st.spinner("Getting response..."):
        response = requests.post("http://localhost:5000/ask", json={"question": user_input})
        if response.status_code == 200:
            answer = response.json().get("response")
            st.session_state.history.append({"question": user_input, "answer": answer})
            st.experimental_rerun()  # Rerun to update the chat history
        else:
            st.error("Error: Unable to get response from the server.")

if st.sidebar.button("Clear History"):
    st.session_state.history = []
    st.experimental_rerun()  # Rerun to clear the chat history
