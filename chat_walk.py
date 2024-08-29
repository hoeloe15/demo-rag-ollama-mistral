import os
import json
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
            logging.debug("Loading conversation state from file.")
            return json.load(file)
    return {"questions": questions, "answers": {}}

def save_conversation_state(state):
    """Save conversation state to a file."""
    with open(conversation_state_file, 'w') as file:
        logging.debug("Saving conversation state to file.")
        json.dump(state, file)

# Load existing conversation state or initialize a new one
conversation_state = load_conversation_state()

# Define the prompt template to dynamically handle questions and answers
conversation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Continue the conversation based on the history. "
            "Acknowledge the user's responses and naturally move on to the next unanswered question. "
            "If a response seems unusual or unclear, ask the user to confirm if that's their final answer. "
            "If all questions are answered, inform the user that they have finished and provide a summary of the answers from the conversation.\n"
            "Questions to be asked:\n{questions}\n"
            "Conversation so far:\n{chat_history}\n"
            "Please ask the next unanswered or not fully answered question from the list to the user."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# Define the prompt template for response evaluation
evaluation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are an assistant evaluating if the user's latest response makes sense based on the question asked. "
            "Please respond with 'go ahead' if the response is appropriate or 'need validation' if it requires further clarification."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# Define the prompt template for confirmation evaluation
confirmation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant confirming if the user's latest response is a confirmation or a denial. "
            "Please respond with 'confirm' if the user's response indicates agreement or confirmation, or 'clarify' if the response indicates a need for further clarification."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# Create RunnableSequences using the pipe operator
conversation_chain = conversation_prompt | model
evaluation_chain = evaluation_prompt | model
confirmation_chain = confirmation_prompt | model

def ask_questions():
    """Main function to manage the conversation."""
    while True:
        # Prepare inputs with the current state of questions and answers
        inputs = {
            "input": "Please continue the conversation.",
            "questions": "\n".join(conversation_state["questions"]),
            "chat_history": memory.load_memory_variables({})['chat_history']
        }

        logging.debug(f"Inputs for conversation chain: {inputs}")

        # Invoke the conversation chain
        output = conversation_chain.invoke(inputs)

        # Extract the content from the AIMessage
        if isinstance(output, AIMessage):
            response = output.content.strip()
        else:
            response = str(output).strip()

        logging.debug(f"Response from conversation chain: {response}")

        print(response)
        user_input = input("Your response (type 'pause' to save and exit): ")

        if user_input.lower() == 'pause':
            logging.debug("User chose to pause the conversation.")
            print("Conversation paused. Your progress has been saved.")
            save_conversation_state(conversation_state)
            break

        # Save the user's response in the conversation history
        memory.save_context({"input": user_input}, {"output": response})

        # Evaluate the user's response with the LLM
        evaluation_input = {
            "input": user_input,
            "chat_history": memory.load_memory_variables({})['chat_history']
        }

        logging.debug(f"Inputs for evaluation chain: {evaluation_input}")

        evaluation_output = evaluation_chain.invoke(evaluation_input)

        # Interpret the evaluation output
        if isinstance(evaluation_output, AIMessage):
            evaluation_result = evaluation_output.content.strip().lower()
        else:
            evaluation_result = str(evaluation_output).strip().lower()

        logging.debug(f"Evaluation result: {evaluation_result}")

        if evaluation_result == 'go ahead':
            # Save answer and continue
            current_question_index = len(conversation_state["answers"])
            if current_question_index < len(conversation_state["questions"]):
                current_question = conversation_state["questions"][current_question_index]
                conversation_state["answers"][current_question] = user_input
                logging.debug(f"Answer saved for question '{current_question}': {user_input}")
        elif evaluation_result == 'need validation':
            # Use the confirmation chain to evaluate the user's clarification
            confirmation_input = {
                "input": user_input,
                "chat_history": memory.load_memory_variables({})['chat_history']
            }

            logging.debug(f"Inputs for confirmation chain: {confirmation_input}")

            confirmation_output = confirmation_chain.invoke(confirmation_input)

            # Interpret the confirmation output
            if isinstance(confirmation_output, AIMessage):
                confirmation_result = confirmation_output.content.strip().lower()
            else:
                confirmation_result = str(confirmation_output).strip().lower()

            logging.debug(f"Confirmation result: {confirmation_result}")

            if confirmation_result == 'confirm':
                # Save answer and continue
                current_question_index = len(conversation_state["answers"])
                if current_question_index < len(conversation_state["questions"]):
                    current_question = conversation_state["questions"][current_question_index]
                    conversation_state["answers"][current_question] = user_input
                    logging.debug(f"Answer saved for question '{current_question}': {user_input}")
            elif confirmation_result == 'clarify':
                print("The LLM needs more information. Please provide a clearer response.")
                continue  # Ask the same question again

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
