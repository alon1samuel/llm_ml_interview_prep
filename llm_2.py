"""
Implement RAG (Retrieval-Augmented Generation) for document retrieval and answering.
Explain how top-p sampling works in LLM output generation.
Use LangChain to integrate an LLM with a vector database for Q&A.
Discuss evaluation strategies for text generation (BLEU, ROUGE, human evaluation).
Demonstrate prompt chaining for a multi-step data extraction process.
Use Faiss to enhance semantic search by integrating with an LLM for context-based ranking.
"""


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# Implement RAG (Retrieval-Augmented Generation) for document retrieval and answering.

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print(len(docs))


print()


from langchain_openai import ChatOpenAI

model = ChatOpenAI()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


question = "What are the approaches to Task Decomposition?"

docs = vectorstore.similarity_search(question)



from langchain_core.runnables import RunnablePassthrough

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)

question = "What are the approaches to Task Decomposition?"

docs = vectorstore.similarity_search(question)

# Run
chain.invoke({"context": docs, "question": question})