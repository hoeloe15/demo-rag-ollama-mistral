import streamlit as st
import requests
from frontend.components import display_chat_history, create_input_form
from frontend.styles import set_custom_styles

st.set_page_config(page_title="Chat with AI", page_icon=":robot_face:", layout="centered")
set_custom_styles()

st.title("AI Chat Interface")

# Initialize session state for history if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = []

# Display history of the conversation
display_chat_history()

# Text input and send button at the bottom
with create_input_form() as form:
    user_input = form.text_area("Your message:", key='input', height=70)
    submit_button = form.form_submit_button(label='Send')

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
