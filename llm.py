"""
Write a prompt to generate summaries of legal patents using an LLM.
Discuss the impact of temperature on LLM output randomness.
Demonstrate how to fine-tune a GPT model on domain-specific data.
Design a prompt for an LLM to classify text into predefined categories.
Implement RAG (Retrieval-Augmented Generation) for document retrieval and answering.
Explain how top-p sampling works in LLM output generation.
Use LangChain to integrate an LLM with a vector database for Q&A.
Discuss evaluation strategies for text generation (BLEU, ROUGE, human evaluation).
Demonstrate prompt chaining for a multi-step data extraction process.
Use Faiss to enhance semantic search by integrating with an LLM for context-based ranking.

Create a production llm 
"""

import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# Write a prompt to generate summaries of legal patents using an LLM.

from langchain_openai import OpenAI

llm = OpenAI()

print(llm.model_name)


from langchain_core.prompts import PromptTemplate

prompt_template = """
Please generate a summary of the following legal document.

Document:
{document}
"""

import polars as pl

legal_docs_df = pl.read_csv('data/mock_patents.csv')
print(legal_docs_df.head())

prompt = PromptTemplate.from_template(prompt_template)

chain = prompt | llm
res = chain.invoke(
    {
        "document": legal_docs_df['Abstract'][0],
    }
)

print(res)
print()


# Discuss the impact of temperature on LLM output randomness.

"""
Temperature is a parameter that decides between precision of the model to creativity of the model. In low 
temperature the model will give answers that are more likely answer the question. Where in high temperature
the model will take lower probability answers and generate them which will be less connected but more creative.
In equation terms - the temperature is in the softmax equation which will determine how sharp logits are turned
into probabilities. temp=0 will produce the delta function.
"""

# Demonstrate how to fine-tune a GPT model on domain-specific data.


from openai import OpenAI
client = OpenAI()

with open('data/mock_chats.jsonl', 'rb') as f:
    file_response = client.files.create(
    file=f,
    purpose="fine-tune"
    )

file_id = file_response.id

# client.fine_tuning.jobs.create(
#   training_file=file_id,
#   model="gpt-4o-mini-2024-07-18",
#   hyperparameters={'n_epochs': 1}
# )


# List 10 fine-tuning jobs

jobs = client.fine_tuning.jobs.list(limit=10)

jobs.data[0].id

# Retrieve the state of a fine-tune

job = client.fine_tuning.jobs.retrieve(
jobs.data[0].id)

print(job)


# Design a prompt for an LLM to classify text into predefined categories.

"""

"""

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_openai import OpenAI

llm = OpenAI()


from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)



from langchain_core.pydantic_v1 import BaseModel, Field



class ClassifyText(BaseModel):
    result: str = Field(description="One of (Positive/Negative/Neutral)")

structured_llm = model.with_structured_output(ClassifyText)


prompt_template = """
Please classify the following text into one of the following: (Positive/Negative/Neutral)
Text:
{text}
"""

prompt = PromptTemplate.from_template(prompt_template)

chain = prompt | structured_llm

sentiment_df = pl.read_csv('data/mock_sentiment.csv')


response = structured_llm.invoke(sentiment_df['News Clip'][0])
print(response.result)
"""
If a few shot examples needed, this can be added using prompt template which can add this.
"""
print()



