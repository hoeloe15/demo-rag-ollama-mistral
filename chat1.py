import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini")

# Define memory to track conversation history with specific memory key
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the evaluation prompt template with a single input key
evaluation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant evaluating the user's response. The conversation is as follows:\n"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}\nIs this a valid response? Please respond with 'yes' or 'no'.")
    ]
)

# Define the explanation prompt for incorrect answers with a single input key
explanation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant providing explanations."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}\nThe response does not seem valid. Please explain why.")
    ]
)

# Define the summary prompt to check if all questions are answered and provide a summary
summary_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant tasked with reviewing the conversation. Based on the conversation history, please perform a gap analysis to check if all questions have been answered. Then, provide a summary of the answers."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Provide a gap analysis and summary.")
    ]
)

# Create chains with memory
evaluation_chain = LLMChain(
    llm=model,
    prompt=evaluation_prompt,
    memory=memory,
    verbose=True
)

explanation_chain = LLMChain(
    llm=model,
    prompt=explanation_prompt,
    memory=memory,
    verbose=True
)

summary_chain = LLMChain(
    llm=model,
    prompt=summary_prompt,
    memory=memory,
    verbose=True
)

def ask_question(question):
    while True:
        # Ask the user for their answer
        user_answer = input(f"{question}\nYour answer: ")

        # Combine question and answer for input handling
        combined_input = f"Question: {question}\nAnswer: {user_answer}"

        # Evaluate the user's answer using the evaluation chain with combined input
        evaluation_output = evaluation_chain.invoke({"input": combined_input})
        evaluation_result = evaluation_output['text'].strip().lower()  # Correctly extract the text

        # Check if the LLM response indicates a correct answer
        if "yes" in evaluation_result:
            print("Correct answer!")
            break
        else:
            # Get an explanation for the incorrect answer
            explanation_output = explanation_chain.invoke({"input": combined_input})['text'].strip()
            print("Incorrect answer, please try again.")
            print(f"Explanation: {explanation_output}")

# List of questions to ask
questions = [
    "What is your name?",
    "What is the name of your company?",
    "How many employees do you have?",
    "Who is responsible for your IT infrastructure at your company?",
    "What is the mission of your company?"
]

# Main loop to ask questions
for question_template in questions:
    ask_question(question_template)

# Generate a summary using the summary chain
def generate_summary():
    summary_output = summary_chain.invoke({"input": ""})['text'].strip()
    print("\nSummary of the conversation:")
    print(summary_output)

generate_summary()
