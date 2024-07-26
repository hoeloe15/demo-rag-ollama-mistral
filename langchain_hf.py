from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the tokenizer and model for Dutch
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = AutoModelForCausalLM.from_pretrained("GroNLP/bert-base-dutch-cased")

# Define the initial prompt
initial_prompt = "Hallo, vandaag gaan we beginnen met het intakegesprek en ga ik je wat vragen stellen om meer over u te weten te komen. Antwoord de vragen zo accuraat mogelijk. U kunt altijd stoppen door 'Ik ben klaar' te typen. Laten we bij het begin beginnen, wat is uw naam?"

# Define the list of questions
questions = [
    "Wat is uw naam?",
    "Wat is uw leeftijd?",
    "Wat is uw beroep?",
    "Waar woont u?",
    "Wat zijn uw hobby's?",
    "Heeft u huisdieren?",
    "Wat is uw favoriete eten?"
]

# Function to get the next question
def get_next_question(index):
    if index < len(questions):
        return questions[index]
    else:
        return "Dank u voor uw antwoorden. Het gesprek is nu ten einde."

# Function to generate a conversational response
def generate_response(model, tokenizer, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Start the conversation
print(initial_prompt)

index = 0
while True:
    # Get user input
    user_input = input("U: ")
    
    # Check for exit condition
    if user_input.lower() == "ik ben klaar":
        print("Chatbot: Bedankt voor uw tijd. Het gesprek is beëindigd.")
        break

    # Prepare the input for the conversation
    if index == 0:
        prompt_text = f"Q: {questions[index]}\nA: {user_input}\n\n"
    else:
        prompt_text = f"Q: {questions[index-1]}\nA: {previous_user_input}\n\n"

    # Generate a conversational response
    response = generate_response(model, tokenizer, prompt_text)

    # Prepare the next question prompt
    next_question = get_next_question(index + 1)
    response_text = f"{response}\n{next_question}"

    # Output the response
    print(f"Chatbot: {response_text}")

    # Store the current user input for the next iteration
    previous_user_input = user_input

    # Increment question index
    index += 1
