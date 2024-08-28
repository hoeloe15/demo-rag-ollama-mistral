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

# Define the explanation prompt for incorrect answers
explanation_prompt = ChatPromptTemplate.from_template(
    "The answer '{answer}' to the question '{question}' is incorrect. Please provide a brief explanation why this is not the right answer."
)

# Define the chains for evaluation and explanation
evaluation_chain = evaluation_prompt | model | StrOutputParser()
explanation_chain = explanation_prompt | model | StrOutputParser()

def ask_question(question):
    while True:
        # Ask the user for their answer
        user_answer = input(f"{question}\nYour answer: ")

        # Evaluate the user's answer
        evaluation_output = evaluation_chain.invoke({"question": question, "answer": user_answer})

        if evaluation_output.strip().lower() == "yes":
            print("Correct answer!")
            break
        else:
            # Get an explanation for the incorrect answer
            explanation_output = explanation_chain.invoke({"question": question, "answer": user_answer})
            print("Incorrect answer, please try again.")
            print(f"Explanation: {explanation_output}")

# List of questions to ask
questions = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Name a programming language that is commonly used for web development."
]

for question in questions:
    ask_question(question)
