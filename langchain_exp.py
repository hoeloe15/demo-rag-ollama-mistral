from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the LLM (ensure you have the correct API keys and packages installed)
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Define the initial prompt
initial_prompt = "Hallo, vandaag gaan we beginnen met het intakegesprek en ga ik je wat vragen stellen om meer over u te weten te komen. Antwoord de vragen zo accuraat mogelijk. U kunt altijd stoppen door 'Ik ben klaar' te typen. Laten we bij het begin beginnen, wat is uw naam?"

# Set up the conversation memory
memory = ConversationBufferMemory()

# Define the template for follow-up questions
question_template = """
{{history}}
Q: {{current_question}}
A: {{user_input}}

{{next_question}}
"""

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["history", "current_question", "user_input", "next_question"],
    template=question_template
)

# Function to get the session history from the memory
def get_session_history(memory):
    return memory.load_memory_variables().get("history", "")

# Initialize the RunnableWithMessageHistory with the memory and prompt template
conversation = RunnableWithMessageHistory(
    runnable=llm,
    get_session_history=lambda: get_session_history(memory)
)

# Define the list of questions
questions = [
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
memory.save_context({"current_question": "Wat is uw naam?"}, {"user_input": ""})

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
        "history": get_session_history(memory),
        "current_question": memory.load_memory_variables().get("current_question", ""),
        "user_input": user_input,
        "next_question": get_next_question(index)
    }

    # Generate the response
    response = conversation.invoke(input_variables)
    
    # Save the conversation state
    memory.save_context({"current_question": get_next_question(index)}, {"user_input": user_input})
    
    # Output the response
    print(f"Chatbot: {response}")

    # Increment question index
    index += 1
