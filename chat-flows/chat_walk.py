import os
import json
import logging
from dotenv import load_dotenv
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Debug flag
DEBUG = False

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

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
                return json.load(file)
    except json.JSONDecodeError:
        logging.error("Failed to load conversation state. Starting a new conversation.")
    return {"questions": questions, "answers": {}}

def save_conversation_state(state):
    """Save conversation state to a file."""
    try:
        with open(conversation_state_file, 'w') as file:
            if DEBUG:
                logging.debug("Saving conversation state to file.")
            json.dump(state, file)
    except IOError:
        logging.error("Failed to save conversation state.")

# Load existing conversation state or initialize a new one
conversation_state = load_conversation_state()

# Define the prompt template
conversation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant having a conversation with a user. Your goal is to collect information based on the provided list of questions.\n\n"
            "Instructions:\n"
            "1. Greet the user and start asking the questions one by one.\n"
            "2. After each question, analyze the user's response to determine if it answers the question satisfactorily.\n"
            "3. If the response does not answer the question, rephrase the question or ask for clarification.\n"
            "4. If the response answers the question, save the answer and move on to the next question.\n"
            "5. If all questions have been answered, provide a summary of the user's responses.\n"
            "6. Be concise and professional in your communication. Use the user's name when addressing them.\n\n"
            "Questions to ask:\n{questions}\n\n"
            "Conversation history:\n{chat_history}\n\n"
            "User: {input}\n"
            "Assistant:"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

def generate_response(user_input):
    """Generate a response based on the conversation history and user input."""
    inputs = {
        "input": user_input,
        "questions": "\n".join(conversation_state["questions"]),
        "chat_history": memory.load_memory_variables({})['chat_history']
    }

    if DEBUG:
        logging.debug(f"Inputs for conversation prompt: {inputs}")

    output = conversation_prompt.format_prompt(**inputs).to_messages()[-1]

    if DEBUG:
        logging.debug(f"Generated response: {output}")

    return output

def ask_questions():
    """Main function to manage the conversation."""
    print("Hello! Let's start our conversation.")

    while True:
        user_input = input("Your response (type 'pause' to save and exit): ")

        if user_input.lower() == 'pause':
            if DEBUG:
                logging.debug("User chose to pause the conversation.")
            print("Conversation paused. Your progress has been saved.")
            save_conversation_state(conversation_state)
            break

        memory.save_context({"input": user_input}, {"output": ""})
        response = generate_response(user_input)

        memory.save_context({"input": ""}, {"output": response.content})
        print(response.content)

        if all(question in conversation_state["answers"] for question in questions):
            if DEBUG:
                logging.debug("All questions have been answered.")
            print("\nThank you for the conversation! Here is a summary of your responses:")
            for question, answer in conversation_state["answers"].items():
                print(f"{question}: {answer}")
            break

        save_conversation_state(conversation_state)

# Start or resume the conversation
ask_questions()