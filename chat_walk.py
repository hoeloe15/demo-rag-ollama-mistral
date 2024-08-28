import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

chain = prompt | model | StrOutputParser()


# print(chain.invoke({"topic": "bears"}))

from langchain_core.output_parsers import StrOutputParser

analysis_prompt = ChatPromptTemplate.from_template("is this a funny joke? {joke}")

composed_chain = {"joke": chain} | analysis_prompt | model | StrOutputParser()

print(composed_chain.invoke({"topic": "bears"}))




# # Initialize memory to keep track of the conversation
# memory = ConversationBufferMemory()

# # Create a function to format the conversation prompt
# def create_conversation_prompt(question):
#     return f"You are a friendly assistant. Ask the user this question: '{question}' and wait for their response."

# # List of questions to ask the user
# questions = [
#     "What's your name?",
#     "Where do you live?",
#     "Where do you work?",
#     # Add more questions as needed
# ]

# # Function to ask questions and store responses
# def ask_questions(llm, questions, memory):
#     for question in questions:
#         # Create a conversation prompt for the current question
#         prompt_text = create_conversation_prompt(question)
#         prompt = ChatPromptTemplate.from_template(prompt_text)
        
#         # Initialize the conversation chain with the current prompt and memory
#         chain = ConversationChain(prompt=prompt, llm=llm, memory=memory)
        
#         # Ask the question
#         print(f"LLM: {question}")
#         user_input = input("Your answer: ")
        
#         # Save user response to memory
#         memory.save_context({"input": question}, {"output": user_input})
        
#         # Process the LLM response to maintain conversation flow
#         response = chain.run({"input": user_input})
#         print(f"LLM: {response}")

# # Start the question-asking process
# ask_questions(llm, questions, memory)
