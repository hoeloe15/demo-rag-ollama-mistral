import streamlit as st
import sys
from styles import apply_custom_css, apply_custom_js
from chat_handler import init_session_state, display_chat_history, handle_user_input, clear_chat_history

def main():
    hosted_local = True if len(sys.argv) > 1 and sys.argv[1] == "local" else False
    backend_url = st.secrets.BACKEND_URL.local_url if hosted_local else st.secrets.BACKEND_URL.backend_url

    apply_custom_css()
    apply_custom_js()
    st.title("AI Chat Interface")
    init_session_state()
    display_chat_history()
    handle_user_input(backend_url)  # Pass backend_url as an argument
    clear_chat_history()

if __name__ == "__main__":
    main()
