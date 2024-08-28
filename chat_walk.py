import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableSequence

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
    if os.path.exists(conversation_state_file):
        with open(conversation_state_file, 'r') as file:
            return json.load(file)
    return {"questions": questions, "answers": {}}

def save_conversation_state(state):
    """Save conversation state to a file."""
    with open(conversation_state_file, 'w') as file:
        json.dump(state, file)

# Load existing conversation state or initialize new one
conversation_state = load_conversation_state()

# Define the prompt template to dynamically handle questions and answers
conversation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant helping the user go through a list of questions one by one. "
            "Based on the conversation so far, decide which question to ask next or check if the answer is valid.\n"
            "Here are the questions:\n{questions}\n"
            "Here are the answers provided so far:\n{answers}\n"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# Create a RunnableSequence
conversation_chain = RunnableSequence(
    steps=[conversation_prompt, model],
    memory=memory
)

def ask_questions():
    """Main function to manage the conversation."""
    while True:
        # Prepare inputs with current state of questions and answers
        inputs = {
            "input": "Please continue the conversation.",
            "questions": "\n".join(conversation_state["questions"]),
            "answers": "\n".join([f"{q}: {a}" for q, a in conversation_state["answers"].items()]),
            "chat_history": memory.load_memory_variables({})['chat_history']
        }

        # Invoke the conversation chain
        output = conversation_chain.invoke(inputs)
        response = output['text'].strip()

        print(response)
        user_input = input("Your response (type 'pause' to save and exit): ")

        if user_input.lower() == 'pause':
            print("Conversation paused. Your progress has been saved.")
            save_conversation_state(conversation_state)
            break

        # Process the LLM's response and user's input
        if response.startswith("Q:"):
            # Extract question and wait for user answer
            question = response[2:].strip()
            conversation_state["answers"][question] = user_input
        elif response.startswith("A:"):
            # Validation or explanation response
            print(response[2:].strip())
            memory.save_context({"input": user_input}, {"output": response})

        # Save the updated state after each interaction
        save_conversation_state(conversation_state)

# Start or resume the conversation
ask_questions()

# Print the conversation history at the end of the script
print("\nConversation History:")
conversation_history = memory.load_memory_variables({})['chat_history']
for message in conversation_history:
    if isinstance(message, HumanMessage):
        print(f"Human: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")
