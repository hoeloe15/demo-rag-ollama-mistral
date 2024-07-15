import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from typing import List
import openai

# Load environment variables from .env file
load_dotenv()

# Configuration
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI LLM
llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Define the prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Je bent een AI language model assistent. Je taak is om zo goed mogelijk de vragen van klanten te beantwoorden met informatie die je uit de toegevoegde data kan vinden in de vectordatabase.
    Je blijft altijd netjes en als je het niet kan vinden in de vectordatabase, geef je dat aan.
    De originele vraag: {question}
    Context: {context}"""
)

# Initialize LLMChain with the prompt template and LLM
chain = LLMChain(
    llm=llm,
    prompt=QUERY_PROMPT
)

def test_chain(question: str, context: str):
    input_data = {"context": context, "question": question}
    response = chain.invoke(input_data)
    print(response)

if __name__ == "__main__":
    # Sample context and question
    sample_context = "Dit is een voorbeeldcontext met relevante informatie."
    sample_question = "Wat is het belangrijkste punt in de context?"

    test_chain(sample_question, sample_context)
