from langchain.prompts import PromptTemplate
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the LLM (ensure you have the correct API keys and packages installed)
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Define the initial prompt
initial_prompt = "Hallo, vandaag gaan we beginnen met het intakegesprek en ga ik je wat vragen stellen om meer over u te weten te komen. Antwoord de vragen zo accuraat mogelijk. U kunt altijd stoppen door 'Ik ben klaar' te typen. Laten we bij het begin beginnen, wat is uw naam?"

# Define the template for follow-up questions
question_template = """
Q: {{current_question}}
A: {{user_input}}

{{next_question}}
"""

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["current_question", "user_input", "next_question"],
    template=question_template
)

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
    input_variables = {
        "current_question": questions[index],
        "user_input": user_input,
        "next_question": get_next_question(index + 1)
    }

    # Generate the response
    try:
        response = llm(prompt.format(**input_variables))
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        break
    
    # Output the response
    print(f"Chatbot: {response}")

    # Increment question index
    index += 1
