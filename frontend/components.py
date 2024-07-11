import streamlit as st

def display_chat_history():
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for item in st.session_state.history:
        st.markdown(f'<div class="message user"><b>You:</b> {item["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="message bot"><b>AI:</b> {item["answer"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_input_form():
    return st.form(key='my_form', clear_on_submit=True)
