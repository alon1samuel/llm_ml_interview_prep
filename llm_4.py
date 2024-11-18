# Use Faiss to enhance semantic search by integrating with an LLM for context-based ranking.

import requests
from io import StringIO
res = requests.get('https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt')
# create dataframe
import pandas as pd
data = pd.read_csv(StringIO(res.text), sep='\t')

sentences = data['sentence_A'].tolist()



from sentence_transformers import SentenceTransformer
# initialize sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')
# create sentence embeddings
sentence_embeddings = model.encode(sentences)
sentence_embeddings.shape

import faiss
d = sentence_embeddings.shape[1]

index = faiss.IndexFlatL2(d)
index.add(sentence_embeddings)


user_query = "Someone sprints with a football"

k = 4
xq = model.encode([user_query])
D, I = index.search(xq, k)  # search
print(I)

vector_query_result = data['sentence_A'].iloc[I[0]].to_dict()
print(vector_query_result)
vector_result_list = [(key, vector_query_result[key]) for key in vector_query_result]

print()



import os
import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# Write a prompt to generate summaries of legal patents using an LLM.

from langchain_openai import OpenAI, ChatOpenAI

model_name = "gpt-4o-mini"
llm = ChatOpenAI(model=model_name)

print(llm.model_name)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(f"Index: {ind} \nContent: {doc}" for ind, doc in docs)

RAG_TEMPLATE = """
You are an assistant for search vector search refinement. 
Use the following pieces of retrieved context and choose the closest to the below query.
The context is mentioned with index and the content. Reply with the index only.

<context>
{context}
</context>

User query:

{user_query}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | llm
    | StrOutputParser()
)

chain_check = RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))

chain

# Run
llm_result = chain.invoke({"context": vector_result_list, "user_query": user_query})

print(f"User query: {user_query}\n")
print(f"vector results list: \n{vector_result_list}\n")
print()

llm_query = data['sentence_A'].iloc[int(llm_result)]
print(llm_query)