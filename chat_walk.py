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
            "You are a helpful assistant. If there is a history, continue the conversation based on the history, "
            "If there is a response, acknowledge the user's responses, and naturally move on to the next unanswered question. "
            "If a response is unclear or incomplete, explain the user why it is unclear for you and ask whether the user is sure and can clarify. Once all questions are answered, summarize the conversation.\n"
            "Questions to be asked:\n{questions}\n"
            "Conversation so far:\n{chat_history}\n"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# Create a RunnableSequence using the pipe operator
conversation_chain = conversation_prompt | model

def ask_questions():
    """Main function to manage the conversation."""
    while True:
        # Prepare inputs with current state of questions and answers
        inputs = {
            "input": "Please continue the conversation.",
            "questions": "\n".join(conversation_state["questions"]),
            "chat_history": memory.load_memory_variables({})['chat_history']
        }

        # Invoke the conversation chain
        output = conversation_chain.invoke(inputs)

        # Extract the content from the AIMessage
        if isinstance(output, AIMessage):
            response = output.content.strip()
        else:
            response = str(output).strip()

        print(response)
        user_input = input("Your response (type 'pause' to save and exit): ")

        if user_input.lower() == 'pause':
            save_conversation_state(conversation_state)
            print("Conversation paused. Your progress has been saved.")
            break

        # Save the user's response in the conversation history
        memory.save_context({"input": user_input}, {"output": response})

        # Check if the LLM response indicates a need for more information or if it confirms understanding
        if "unclear" in response.lower() or "could you clarify" in response.lower() or "I'm not sure I understood" in response.lower():
            print("The LLM needs more information. Please provide a clearer response.")
        else:
            # The LLM has acknowledged a valid answer and moved on
            current_question_index = len(conversation_state["answers"])
            if current_question_index < len(conversation_state["questions"]):
                current_question = conversation_state["questions"][current_question_index]
                conversation_state["answers"][current_question] = user_input
        
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
