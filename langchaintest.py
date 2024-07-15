import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.schema import Document

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

# Create a RunnableSequence
sequence = RunnableSequence(
    steps=[
        QUERY_PROMPT,  # Use the prompt template
        llm  # Pass the processed input to the LLM
    ]
)

def test_chain(question: str, context: str):
    input_data = {"context": context, "question": question}
    response = sequence.invoke(input_data)
    print(response)

if __name__ == "__main__":
    # Sample context and question
    sample_context = "Dit is een voorbeeldcontext met relevante informatie."
    sample_question = "Wat is het belangrijkste punt in de context?"

    test_chain(sample_question, sample_context)
