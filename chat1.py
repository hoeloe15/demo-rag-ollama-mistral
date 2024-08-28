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
    "Is the following answer a valid response for the question provided? Question: '{question}' Answer: '{answer}'. Please respond with 'yes' or 'no' only."
)

# Define the explanation prompt for incorrect answers
explanation_prompt = ChatPromptTemplate.from_template(
    "The answer '{answer}' to the question '{question}' does not seem like a valid response. Please provide a brief explanation of why this is not the right answer."
)

# Define the chains for evaluation and explanation
evaluation_chain = evaluation_prompt | model | StrOutputParser()
explanation_chain = explanation_prompt | model | StrOutputParser()

def ask_question(question):
    while True:
        # Ask the user for their answer
        user_answer = input(f"{question}\nYour answer: ")

        # Evaluate the user's answer
        evaluation_output = evaluation_chain.invoke({"question": question, "answer": user_answer}).strip().lower()

        # Check if the LLM response indicates a correct answer
        if "yes" in evaluation_output:
            print("Correct answer!")
            break
        else:
            # Get an explanation for the incorrect answer
            explanation_output = explanation_chain.invoke({"question": question, "answer": user_answer}).strip()
            print("Incorrect answer, please try again.")
            print(f"Explanation: {explanation_output}")

# List of questions to ask
questions = [
    "What is your name?",
    "What is the name of your company?",
    "How many employees do you have?",
    "Who is responsible for your IT infrastructure?",
    "What is the mission of your company?"
]

for question in questions:
    ask_question(question)
