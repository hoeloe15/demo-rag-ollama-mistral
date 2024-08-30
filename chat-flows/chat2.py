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

# Define the prompt template to dynamically handle questions and answers
conversation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Continue the conversation based on the history. "
            "Acknowledge the user's responses and naturally move on to the next unanswered question. "
            "If a response seems unusual or unclear, ask the user to confirm if that's their final answer. "
            "If all questions are answered, inform the user that they have finished and provide a summary of the answers from the conversation.\n"
            "- Respond in a friendly and conversational tone.\n"
            "- Use the user's name when addressing them.\n"
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
            "- Check if the response is relevant to the question asked.\n"
            "- Verify if the response contains the required information.\n"
            "- Be lenient with short answers and only request validation for ambiguous or unclear responses.\n"
            "Please respond with 'go ahead' if the response is appropriate or 'need validation' if it requires further clarification."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# Define the prompt template for clarification and confirmation
confirmation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant seeking confirmation from the user. "
            "Please respond with a natural confirmation question based on the user's latest response. "
            "For example, if the user's name is Mark, you can ask: 'Just to confirm, your name is Mark. Is that correct?'"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

confirmation_evaluation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are an assistant evaluating if the user's latest response indicates a confirmation or not. "
            "Please respond with 'confirmed' if the response seems to be a confirmation, or 'not confirmed' if it doesn't."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# Create RunnableSequences using the pipe operator
conversation_chain = conversation_prompt | model
evaluation_chain = evaluation_prompt | model
confirmation_chain = confirmation_prompt | model
confirmation_evaluation_chain = confirmation_evaluation_prompt | model

def validate_user_input(user_input):
    """Validate user input."""
    if not user_input.strip():
        raise ValueError("User input cannot be empty.")

def get_random_greeting():
    """Get a random greeting."""
    greetings = ["Hello", "Hi", "Hey", "Greetings"]
    return random.choice(greetings)

def get_random_acknowledgement():
    """Get a random acknowledgement."""
    acknowledgements = ["I see", "Okay", "Alright", "Understood"]
    return random.choice(acknowledgements)

def ask_questions():
    """Main function to manage the conversation."""
    while True:
        try:
            # Prepare inputs with the current state of questions and answers
            inputs = {
                "input": "Please continue the conversation.",
                "questions": "\n".join(conversation_state["questions"]),
                "chat_history": memory.load_memory_variables({})['chat_history']
            }

            if DEBUG:
                logging.debug(f"Inputs for conversation chain: {inputs}")

            # Invoke the conversation chain
            output = conversation_chain.invoke(inputs)

            # Extract the content from the AIMessage
            if isinstance(output, AIMessage):
                response = output.content.strip()
            else:
                response = str(output).strip()

            if DEBUG:
                logging.debug(f"Response from conversation chain: {response}")

            print(response)
            user_input = input("Your response (type 'pause' to save and exit): ")

            if user_input.lower() == 'pause':
                if DEBUG:
                    logging.debug("User chose to pause the conversation.")
                print("Conversation paused. Your progress has been saved.")
                save_conversation_state(conversation_state)
                break

            validate_user_input(user_input)

            # Save the user's response in the conversation history
            memory.save_context({"input": user_input}, {"output": response})

            # Evaluate the user's response with the LLM
            evaluation_input = {
                "input": user_input,
                "chat_history": memory.load_memory_variables({})['chat_history']
            }

            if DEBUG:
                logging.debug(f"Inputs for evaluation chain: {evaluation_input}")

            evaluation_output = evaluation_chain.invoke(evaluation_input)

            # Interpret the evaluation output
            if isinstance(evaluation_output, AIMessage):
                evaluation_result = evaluation_output.content.strip().lower()
            else:
                evaluation_result = str(evaluation_output).strip().lower()

            if DEBUG:
                logging.debug(f"Evaluation result: {evaluation_result}")

            if evaluation_result == 'go ahead':
                # Save answer and continue
                current_question_index = len(conversation_state["answers"])
                if current_question_index < len(conversation_state["questions"]):
                    current_question = conversation_state["questions"][current_question_index]
                    conversation_state["answers"][current_question] = user_input
                    if DEBUG:
                        logging.debug(f"Answer saved for question '{current_question}': {user_input}")
                    print(get_random_acknowledgement() + ". Let's move on.")
            elif evaluation_result == 'need validation':
                # Use the confirmation chain to generate a natural confirmation question
                confirmation_input = {
                    "input": user_input,
                    "chat_history": memory.load_memory_variables({})['chat_history']
                }

                if DEBUG:
                    logging.debug(f"Inputs for confirmation chain: {confirmation_input}")

                confirmation_output = confirmation_chain.invoke(confirmation_input)

                # Extract the confirmation question from the output
                if isinstance(confirmation_output, AIMessage):
                    confirmation_question = confirmation_output.content.strip()
                else:
                    confirmation_question = str(confirmation_output).strip()

                print(confirmation_question)
                user_confirmation = input("Your response (type 'pause' to save and exit): ")

                if user_confirmation.lower() == 'pause':
                    if DEBUG:
                        logging.debug("User chose to pause the conversation.")
                    print("Conversation paused. Your progress has been saved.")
                    save_conversation_state(conversation_state)
                    return

                # Evaluate the user's confirmation with the LLM
                confirmation_evaluation_input = {
                    "input": user_confirmation,
                    "chat_history": memory.load_memory_variables({})['chat_history']
                }

                if DEBUG:
                    logging.debug(f"Inputs for confirmation evaluation chain: {confirmation_evaluation_input}")

                confirmation_evaluation_output = confirmation_evaluation_chain.invoke(confirmation_evaluation_input)

                # Interpret the confirmation evaluation output
                if isinstance(confirmation_evaluation_output, AIMessage):
                    confirmation_evaluation_result = confirmation_evaluation_output.content.strip().lower()
                else:
                    confirmation_evaluation_result = str(confirmation_evaluation_output).strip().lower()

                if DEBUG:
                    logging.debug(f"Confirmation evaluation result: {confirmation_evaluation_result}")

                if confirmation_evaluation_result == 'confirmed':
                    # Save answer and continue
                    current_question_index = len(conversation_state["answers"])
                    if current_question_index < len(conversation_state["questions"]):
                        current_question = conversation_state["questions"][current_question_index]
                        conversation_state["answers"][current_question] = user_input
                        if DEBUG:
                            logging.debug(f"Answer saved for question '{current_question}': {user_input}")
                        print(get_random_acknowledgement() + ". Let's proceed.")
                else:
                    print("I apologize for the confusion. Let's try that again.")
                    continue

            # Save the updated state after each interaction
            save_conversation_state(conversation_state)
        except ValueError as e:
            logging.error(f"Error: {str(e)}")
            print("Oops! It seems like there was an issue with your input. Please try again.")


# Start or resume the conversation
print(get_random_greeting() + "! Let's start our conversation.")
ask_questions()

# Print the conversation history at the end of the script
print("\nConversation History:")
conversation_history = memory.load_memory_variables({})['chat_history']
for message in conversation_history:
    if isinstance(message, HumanMessage):
        print(f"Human: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")