import streamlit as st
import requests

def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []

def display_chat_history():
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for item in st.session_state.history:
        st.markdown(f'<div class="message user"><b>You:</b> {item["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="message bot"><b>AI:</b> {item["answer"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def handle_user_input(backend_url):
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your message:", key='input', height=70)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        with st.spinner("Getting response..."):
            response = requests.post(f"{backend_url}/ask", json={"question": user_input})
            if response.status_code == 200:
                answer = response.json().get("response")
                st.session_state.history.append({"question": user_input, "answer": answer})
                st.experimental_rerun()  # Rerun to update the chat history
            else:
                st.error("Error: Unable to get response from the server.")

def clear_chat_history():
    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()  # Rerun to clear the chat history
