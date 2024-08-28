import logging
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
logger.info("Loaded environment variables.")

# Initialize the OpenAI LLM
logger.info("Initializing the OpenAI LLM.")
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo") 
logger.info("LLM initialized.")

# Set up the conversation memory
memory = ConversationBufferMemory()

# Define the initial prompt template
initial_prompt = "Goedemorgen, vandaag gaan we wat informatie verzamelen. Mag ik beginnen met uw naam?"

# Define the template for follow-up questions
question_template = """
Je antwoord op de vraag '{previous_question}' was '{user_input}'. Dank je wel, {user_name}!
{next_question}
"""

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["previous_question", "user_input", "user_name", "next_question"],
    template=question_template
)

# Create a chat prompt template from the prompt template
chat_prompt = ChatPromptTemplate.from_prompt_template(prompt_template)

# Define the list of questions
questions = [
    "Wat is uw geboortedatum?",
    "Waar woont u?",
    "Wat zijn uw hobby's?",
    "Wat is uw favoriete eten?",
]

def get_next_question(index):
    if index < len(questions):
        return questions[index]
    else:
        return "Dank u voor uw antwoorden. Het gesprek is nu ten einde."

# Start the conversation
print(initial_prompt)
logger.info("Initial prompt sent to user.")

index = 0
previous_question = "Wat is uw naam?"
previous_user_input = ""

while True:
    # Get user input
    user_input = input("U: ")
    logger.info(f"User input: {user_input}")

    # Check for exit condition
    if user_input.lower() == "ik ben klaar":
        print("Chatbot: Bedankt voor uw tijd. Het gesprek is beëindigd.")
        logger.info("Conversation ended by user.")
        break

    # Capture the user's name from the first input
    if index == 0:
        user_name = user_input
    else:
        user_name = memory.load_memory_variables().get("user_name", "gebruiker")

    # Get the next question
    next_question = get_next_question(index)

    # Generate a dynamic conversational prompt
    input_variables = {
        "previous_question": previous_question,
        "user_input": user_input,
        "user_name": user_name,
        "next_question": next_question
    }

    logger.info(f"Input variables for LLM: {input_variables}")

    # Create the chain using LCEL
    chain = (
        {"previous_question": RunnablePassthrough(), "user_input": RunnablePassthrough(), "user_name": RunnablePassthrough(), "next_question": RunnablePassthrough()}
        | chat_prompt
        | llm
        | StrOutputParser()
    )

    # Generate a conversational response
    response = chain.invoke(input_variables)

    # Output the response
    print(f"Chatbot: {response}")

    # Update memory
    memory.save_context({"previous_question": previous_question, "user_input": user_input})

    # Store the current user input and question for the next iteration
    previous_user_input = user_input
    previous_question = next_question

    # Increment question index
    index += 1
    logger.info(f"Next question index: {index}")
