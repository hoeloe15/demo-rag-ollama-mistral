import os
import json
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Debug flag
DEBUG = False

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini")

# Define memory to track conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# List of questions to ask
questions = [
    "What is your name?",
    "What is the name of your company?",
    "How many employees do you have?",
    "Who is responsible for your IT infrastructure at your company?",
    "What is the mission of your company?"
]

# Path for saving and loading conversation state
conversation_state_file = "conversation_state.json"

def load_conversation_state():
    """Load conversation state from a file."""
    try:
        if os.path.exists(conversation_state_file):
            with open(conversation_state_file, 'r') as file:
                if DEBUG:
                    logging.debug("Loading conversation state from file.")
                state = json.load(file)
                if "current_question_index" not in state:
                    state["current_question_index"] = 0
                return state
    except json.JSONDecodeError:
        logging.error("Failed to load conversation state. Starting a new conversation.")
    return {"questions": questions, "answers": {}, "current_question_index": 0}

def save_conversation_state(state):
    """Save conversation state to a file."""
    try:
        with open(conversation_state_file, 'w') as file:
            if DEBUG:
                logging.debug("Saving conversation state to file.")
            json.dump(state, file)
    except IOError:
        logging.error("Failed to save conversation state.")

def initialize_conversation():
    """Initialize the conversation with the list of questions."""
    initial_prompt = f"Hello! I'm here to collect some information from you about you and your company.\nIf you type 'pause', you will save and exit the program, if you type 'finish' you will save and end the conversation. I'll be asking you the following questions:\n\n{chr(10).join(questions)}\n\nLet's get started!\n\nCan you please tell me your name?"
    memory.save_context({"input": ""}, {"output": initial_prompt})
    return initial_prompt

def generate_response(user_input, chat_history):
    """Generate a response based on the conversation history and user input."""
    inputs = {
        "input": user_input,
        "questions": "\n".join(questions),
        "chat_history": "\n".join([msg['content'] for msg in chat_history])
    }

    prompt = (
        "You are a helpful assistant having a conversation with a user. Your goal is to collect information based on the provided list of questions.\n\n"
        "Instructions:\n"
        "1. Analyze the conversation history to determine which questions have been answered satisfactorily.\n"
        "2. If there are unanswered questions, ask the next question in a natural way, building upon the previous conversation.\n"
        "3. If the user's response does not directly answer the question, rephrase the question or ask for clarification. Only do this once per question to avoid annoying the user.\n"
        "4. If the user provides a satisfactory answer, save the answer and move on to the next question.\n"
        "5. If all questions have been answered, provide a summary of the user's responses and ask if everything is correct and end the program by typing 'finish' or if something needs to be changed.\n"
        "6. If the user agrees with the summary and nothing needs to change, ask them to type finish.\n"
        "7. Be concise and professional in your communication. Use the user's name when addressing them.\n\n"
        "Questions to ask:\n{questions}\n\n"
        "Conversation history:\n{chat_history}\n\n"
        "User: {input}\n"
        "Assistant:"
    )

    if DEBUG:
        logging.debug(f"Inputs for conversation prompt: {inputs}")

    # Updated call to use invoke method
    response = model.invoke(prompt.format(**inputs))

    if DEBUG:
        logging.debug(f"Generated response: {response.content}")

    return response.content
