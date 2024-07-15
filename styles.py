import streamlit as st

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
