import streamlit as st
from chat_bot import initialize_conversation, generate_response, load_conversation_state, save_conversation_state

# Custom CSS for improved styling
st.markdown("""
    <style>
        .main-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .chat-box {
            background-color: #333;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border: 1px solid #444;
            color: white;
        }
        .chat-box p {
            margin: 0;
            padding: 5px 0;
        }
        .chat-user {
            text-align: right;
            color: white;
        }
        .chat-assistant {
            text-align: left;
            color: white;
        }
        .input-container {
            margin-top: 20px;
        }
        .stTextInput>div>div>input {
            color: white;  /* Make input text white */
        }
        .stTextInput>div>div>input:focus {
            background-color: #333;  /* Input background same as chat box */
        }
        .stButton>button {
            width: 100%;
            padding: 10px;
            border-radius: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            margin-top: 5px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

def ask_questions():
    """Main function to manage the conversation."""
    if 'conversation_state' not in st.session_state:
        st.session_state['conversation_state'] = load_conversation_state()
        st.session_state['conversation_state']['chat_history'] = []

    st.title("Introductie gesprek")

    # Initialize conversation
    if 'initialized' not in st.session_state:
        initial_prompt = initialize_conversation()
        st.session_state['conversation_state']['chat_history'].append({"role": "assistant", "content": initial_prompt})
        st.session_state['initialized'] = True

    # Display chat history in one consolidated chat box
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)

    for message in st.session_state['conversation_state']['chat_history']:
        if message['role'] == 'user':
            st.markdown(f'<p class="chat-user"><strong>User:</strong> {message["content"]}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="chat-assistant"><strong>Assistant:</strong> {message["content"]}</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-box
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

    # Input box and submit button within a form to enable Enter key submission
    with st.form(key='input_form', clear_on_submit=True):
        user_input = st.text_input("Your response:", key="user_input", label_visibility="collapsed")
        submit_button = st.form_submit_button('Submit')

    if submit_button and user_input:
        if user_input.lower() == 'pause':
            st.write("Conversation paused. Your progress has been saved.")
            save_conversation_state(st.session_state['conversation_state'])
        elif user_input.lower() == 'finish':
            response = generate_response(user_input, st.session_state['conversation_state']['chat_history'])
            st.session_state['conversation_state']['chat_history'].append({"role": "user", "content": user_input})
            st.session_state['conversation_state']['chat_history'].append({"role": "assistant", "content": response})
            st.write(response)
            st.write("\nThank you for the conversation! Here is a summary of your responses:")
            for question in st.session_state['conversation_state']["questions"]:
                answer = st.session_state['conversation_state']['answers'].get(question, "Not answered")
                st.write(f"{question}: {answer}")
            save_conversation_state(st.session_state['conversation_state'])
        else:
            response = generate_response(user_input, st.session_state['conversation_state']['chat_history'])
            st.session_state['conversation_state']['chat_history'].append({"role": "user", "content": user_input})
            st.session_state['conversation_state']['chat_history'].append({"role": "assistant", "content": response})
            save_conversation_state(st.session_state['conversation_state'])
            st.rerun()  # To refresh the chat history and show new messages

# Start the conversation
ask_questions()
