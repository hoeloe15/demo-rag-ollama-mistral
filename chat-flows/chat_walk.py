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
            "5. If all questions have been answered, provide a summary of the user's responses and ask the user to validate it. If the user is happy with the results, they should type 'finish' to end the conversation.\n"
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

def initialize_conversation():
    """Initialize the conversation with the list of questions."""
    initial_prompt = f"Hello! I'm here to collect some information from you about you and your company.\nIf you type 'pause', you will save and exit the program, if you type 'finish' you will save and end the conversation. I'll be asking you the following questions:\n\n{chr(10).join(questions)}\n\nLet's get started!\n\nCan you please tell me your name?"
    memory.save_context({"input": ""}, {"output": initial_prompt})
    print(initial_prompt)

def generate_response(user_input):
    """Generate a response based on the conversation history and user input."""
    inputs = {
        "input": user_input,
        "questions": "\n".join(questions),
        "chat_history": memory.load_memory_variables({})['chat_history']
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

    output = model(prompt.format(**inputs)).content

    if DEBUG:
        logging.debug(f"Generated response: {output}")

    return output

def ask_questions():
    """Main function to manage the conversation."""
    initialize_conversation()


    while True:
        user_input = input("Your response: ")

        if user_input.lower() == 'pause':
            if DEBUG:
                logging.debug("User chose to pause the conversation.")
            print("Conversation paused. Your progress has been saved.")
            save_conversation_state(conversation_state)
            break

        if user_input.lower() == 'finish':
            if DEBUG:
                logging.debug("User chose to end the conversation.")
            memory.save_context({"input": user_input}, {"output": ""})
            response = generate_response(user_input)
            memory.save_context({"input": ""}, {"output": response})
            print(response)
            print("\nThank you for the conversation! Here is a summary of your responses:")
            for question in questions:
                answer = next((a for q, a in conversation_state["answers"].items() if q == question), "Not answered")
                print(f"{question}: {answer}")
            break

        memory.save_context({"input": user_input}, {"output": ""})
        response = generate_response(user_input)

        memory.save_context({"input": ""}, {"output": response})
        print(response)

        save_conversation_state(conversation_state)

# Start or resume the conversation
ask_questions()