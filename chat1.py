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

        # Combine question and answer for input handling
        combined_input = f"Question: {question}\nAnswer: {user_answer}"

        # Debug: Print the memory content before evaluation
        print("Memory before evaluation:", memory.load_memory_variables({}))

        # Evaluate the user's answer using the evaluation chain with combined input
        evaluation_output = evaluation_chain.invoke({"input": combined_input})
        print(evaluation_output)
        evaluation_result = evaluation_output.strip().lower()

        # Debug: Print the evaluation output
        print("Evaluation output:", evaluation_output)

        # Check if the LLM response indicates a correct answer
        if "yes" in evaluation_result:
            print("Correct answer!")
            # Manually save the context to memory as a single input-output pair
            memory.save_context({"input": combined_input}, {"output": evaluation_result})
            break
        else:
            # Get an explanation for the incorrect answer
            explanation_output = explanation_chain.invoke({"input": combined_input}).strip()
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
        chat_history = memory.load_memory_variables({}).get('chat_history', [])
        company_name = "your company"
        # Extract company name from chat history if available
        for message in chat_history:
            if "company" in message.content.lower():
                company_name = message.content.split("company:")[-1].strip()
                break
        question = question_template.format(company_name=company_name)
    else:
        question = question_template
    ask_question(question)

print("Conversation complete! Here is what we learned:")
print(memory.load_memory_variables({}))
