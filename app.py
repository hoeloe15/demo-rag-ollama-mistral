import streamlit as st
from styles import apply_custom_css, apply_custom_js
from chat_handler import init_session_state, display_chat_history, handle_user_input, clear_chat_history

def main():
    apply_custom_css()
    apply_custom_js()
    st.title("AI Chat Interface")
    init_session_state()
    display_chat_history()
    
    # Get the backend URL from Streamlit secrets
    backend_url = st.secrets["BACKEND_URL"]
    handle_user_input(backend_url)
    
    clear_chat_history()

if __name__ == "__main__":
    main()
