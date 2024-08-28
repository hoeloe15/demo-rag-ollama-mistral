import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini")

# Define the evaluation prompt
evaluation_prompt = ChatPromptTemplate.from_template(
    "Is this a valid answer to the question '{question}': '{answer}'? Please respond with 'yes' or 'no'."
)

# Define the chain for evaluation
evaluation_chain = evaluation_prompt | model | StrOutputParser()

def ask_question(question):
    # Ask the user for their answer
    user_answer = input(f"{question}\nYour answer: ")

    # Evaluate the user's answer
    evaluation_output = evaluation_chain.invoke({"question": question, "answer": user_answer})
    
    return evaluation_output.strip().lower() == "yes"

# List of questions to ask
questions = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Name a programming language that is commonly used for web development."
]

for question in questions:
    if ask_question(question):
        print("Correct answer!")
    else:
        print("Incorrect answer, please try again.")
        # Optionally, you could add logic to re-ask the question or exit the loop
