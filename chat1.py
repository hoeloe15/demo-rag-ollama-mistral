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

# Define memory to track conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the evaluation prompt template
evaluation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant evaluating the user's answer to the question. The question: {question}\n"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Answer: {answer}\nIs this a valid response? Please respond with 'yes' or 'no'.")
    ]
)

# Define the explanation prompt for incorrect answers
explanation_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant providing explanations."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("The answer '{answer}' to the question '{question}' does not seem like a valid response. Please provide a brief explanation of why this is not the right answer.")
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

def ask_question(question):
    while True:
        # Ask the user for their answer
        user_answer = input(f"{question}\nYour answer: ")

        # Evaluate the user's answer using the evaluation chain
        evaluation_output = evaluation_chain.invoke({"question": question, "answer": user_answer}).strip().lower()

        # Check if the LLM response indicates a correct answer
        if "yes" in evaluation_output:
            print("Correct answer!")
            # Update memory with the user's answer
            memory.chat_memory.add_user_message(f"Question: {question}\nYour answer: {user_answer}")
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
    "Who is responsible for your IT infrastructure at {company_name}?",
    "What is the mission of {company_name}?"
]

# Main loop to ask questions
for question_template in questions:
    # Generate the question dynamically with memory
    if '{company_name}' in question_template:
        company_name = memory.chat_memory.messages[-1].content.split(":")[-1].strip() if 'company' in memory.chat_memory.messages[-1].content else "your company"
        question = question_template.format(company_name=company_name)
    else:
        question = question_template
    ask_question(question)

print("Conversation complete! Here is what we learned:")
print(memory.load_memory_variables({}))
