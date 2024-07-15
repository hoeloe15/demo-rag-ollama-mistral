import streamlit as st
import requests
import os

def main():
    st.title("AI Chat Interface")
    backend_url = os.getenv("BACKEND_URL", "http://localhost:5000")

    # Your Streamlit app code here
    # Use backend_url for API requests

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your message:", key='input', height=70)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        with st.spinner("Getting response..."):
            response = requests.post(f"{backend_url}/ask", json={"question": user_input})
            if response.status_code == 200:
                answer = response.json().get("response")
                st.session_state.history.append({"question": user_input, "answer": answer})
                st.experimental_rerun()
            else:
                st.error("Error: Unable to get response from the server.")

    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
