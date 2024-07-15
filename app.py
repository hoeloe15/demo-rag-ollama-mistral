import streamlit as st
import requests

# Streamlit app configuration
st.set_page_config(page_title="Chat with AI", page_icon=":robot_face:", layout="centered")

# Custom CSS for chat interface
def apply_custom_css():
    st.markdown("""
        <style>
        body {
            background-color: #2c2f33;
            color: #ffffff;
        }
        .chat-box {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #4a4a4a;
            border-radius: 8px;
            background-color: #23272a;
            color: #ffffff;
        }
        .message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            color: #ffffff;
        }
        .user {
            background-color: #7289da;
            text-align: right;
        }
        .bot {
            background-color: #99aab5;
        }
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #2c2f33;
            padding: 10px 20px;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        }
        .input-container textarea {
            width: 80%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #4a4a4a;
            resize: none;
            background-color: #23272a;
            color: #ffffff;
        }
        .input-container button {
            width: 18%;
            padding: 10px;
            border: none;
            border-radius: 8px;
            background-color: #7289da;
            color: white;
            cursor: pointer;
        }
        .input-container button:hover {
            background-color: #5b6eae;
        }
        </style>
    """, unsafe_allow_html=True)

# Custom JavaScript for handling Enter and Shift+Enter keys
def apply_custom_js():
    st.markdown("""
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const textArea = document.querySelector('textarea');
            textArea.addEventListener('keydown', function(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    document.querySelector('button[type="submit"]').click();
                }
            });
        });
        </script>
    """, unsafe_allow_html=True)

# Initialize session state for history
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []

# Display chat history
def display_chat_history():
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for item in st.session_state.history:
        st.markdown(f'<div class="message user"><b>You:</b> {item["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="message bot"><b>AI:</b> {item["answer"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Handle user input and response
def handle_user_input():
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

# Clear chat history
def clear_chat_history():
    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()  # Rerun to clear the chat history

def main():
    apply_custom_css()
    apply_custom_js()
    st.title("AI Chat Interface")
    init_session_state()
    display_chat_history()
    handle_user_input()
    clear_chat_history()

if __name__ == "__main__":
    main()
