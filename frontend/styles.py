import streamlit as st

def set_custom_styles():
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
