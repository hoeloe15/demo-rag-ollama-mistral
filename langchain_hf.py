import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
logger.info("Loaded environment variables.")

# Initialize the tokenizer and model for Dutch
logger.info("Initializing the tokenizer and model for Dutch.")
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = AutoModelForCausalLM.from_pretrained("GroNLP/bert-base-dutch-cased")
logger.info("Tokenizer and model initialized.")

# Define the initial prompt
initial_prompt = "Goedemorgen, vandaag gaan we wat informatie verzamelen. Mag ik beginnen met uw naam?"

# Define the list of questions (starting from the second question)
questions = [
    "Wat is uw geboortedatum?",
    "Waar woont u?",
    "Wat zijn uw hobby's?",
    "Wat is uw favoriete eten?",
]

# Function to get the next question
def get_next_question(index):
    if index < len(questions):
        return questions[index]
    else:
        return "Dank u voor uw antwoorden. Het gesprek is nu ten einde."

# Function to generate a conversational response
def generate_response(model, tokenizer, prompt_text):
    logger.info(f"Generating response for prompt: {prompt_text}")
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=150, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated response: {response}")
    return response

# Function to create a conversational prompt
def get_conversational_prompt(user_name, previous_question, user_input, next_question):
    return f"Je antwoord op de vraag '{previous_question}' was '{user_input}'. Dank je wel, {user_name}! {next_question}"

# Start the conversation
print(initial_prompt)
logger.info("Initial prompt sent to user.")

user_name = ""
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
        next_question = get_next_question(index)
        prompt_text = f"Goedemorgen {user_name}, wat is uw geboortedatum?"
    else:
        # Generate a dynamic conversational prompt
        next_question = get_next_question(index)
        prompt_text = f"Je antwoord op de vraag '{previous_question}' was '{previous_user_input}'. Dank je wel, {user_name}! {next_question}"
    
    logger.info(f"Prompt text for LLM: {prompt_text}")
    # Generate a conversational response
    response = generate_response(model, tokenizer, prompt_text)

    # Output the response
    print(f"Chatbot: {response}")

    # Store the current user input and question for the next iteration
    previous_user_input = user_input
    previous_question = next_question

    # Increment question index
    index += 1
    logger.info(f"Next question index: {index}")
