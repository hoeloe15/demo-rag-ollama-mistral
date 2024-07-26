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
    prompt_text = f"Q: {questions[index]}\nA: {user_input}\n\n{get_next_question(index + 1)}"
    inputs = tokenizer(prompt_text, return_tensors="pt")

    # Generate the response
    try:
        outputs = model.generate(inputs["input_ids"])
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during model invocation: {e}")
        break
    
    # Output the response
    print(f"Chatbot: {response}")

    # Increment question index
    index += 1
