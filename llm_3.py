# Words with faiss env

"""
Explain how top-p sampling works in LLM output generation.
Use LangChain to integrate an LLM with a vector database for Q&A.
Discuss evaluation strategies for text generation (BLEU, ROUGE, human evaluation).
Demonstrate prompt chaining for a multi-step data extraction process.
Use Faiss to enhance semantic search by integrating with an LLM for context-based ranking.
"""

# 
# Explain how top-p sampling works in LLM output generation.
"""
An llm is producing probabilities for the tokens that represent words. It then chooses the next word.
For top-p parameter it selects the most likely words that make up to p precent probability, then chooses 
a random word from them. This creates different way to produce creative results that are still connected 
to the model input and previous output. 
"""

# 
# Use LangChain to integrate an LLM with a vector database for Q&A.
# Changed to -> convert a qna database into a vector database

qna_path = "data/SQuAD2.0-train-v2.0.json"
import json
with open(qna_path) as f:
    qna_database = json.load(f)

debug = True
if debug:
    qna_database = qna_database['data'][:3]

# Database has subjects, each one has paragraphs, each has questions and each has possible answers
# (qna_database['data'][0]['paragraphs'][0]['qas'][0])

# We'll vectorise only the questions for fast search and then give back the context from it. 

questions_info_list = []
for topic in qna_database:
    for paragraph in topic['paragraphs']:
        for question in paragraph['qas']:
            questions_info_list.append([question['id'], question['question'], question['answers']])

import polars as pl
questions_info_df = pl.DataFrame(questions_info_list, schema=['id', 'question', 'answers'], orient='row')
if debug: questions_info_df = questions_info_df.sample(400)

questions_only_list = questions_info_df['question'].to_list()

from sentence_transformers import SentenceTransformer
# initialize sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')
# create sentence embeddings
questions_embeddings = model.encode(questions_only_list)

import faiss
d = questions_embeddings.shape[1]

index = faiss.IndexFlatL2(d)
index.add(questions_embeddings)


k = 10
query_question = "Where did beyonce grow up?"
xq = model.encode([query_question])
D, I = index.search(xq, k)  # search
# print(I)
# [print(q) for q in questions_info_df['question'][I[0]]]

#

# Discuss evaluation strategies for text generation (BLEU, ROUGE, human evaluation).

"""
rouge is matching the overlapping of words from the reference text (which probably is human annotated) 
to the generated text. It can be more complicated with rouge-2 only matching words pairs (rouge-3 for
3 words sequences and so on) and rouge-s allowing for words pairs with some skips in the middle.
blue is a metric that also compares the machine generation output to a human evaluation (originally
for translation tasks). It actually says in precentage - "How many n-gram words from the machine output can
be found in the human reference?" Usually it goes from 1-gram to 4-gram and multiply them together. blue-1 
for 1-gram is finding the how many words are from the machine output are in the reference. blue-2 is for
2 gram which is words pairs, and so on. Then the precisions are multiplied. 

Both of these metrics are using human annotations as reference to each machine generation. They can't 
work on their own. Another option is for them to be compared to themselves, but only for similarity. 

Blue is giving 1 score for the precision and is widely used in the literature because it is easy to 
compare different results. Rouge can give 2 scores - recall and precision and gives a bit more details
on the result. They capture different aspects of the machine generation, for example rouge-s 2 can capture
word pairs that are present but with skips which allows for more machine freedom in the generation while 
blue doesn't. That makes rouge slower to compute as more calculations are needed for skips considerations. 
"""

# 
# Demonstrate prompt chaining for a multi-step data extraction process.


prompts = [
    "Summarize the key trends in global temperature changes over the past century.",
"Based on the trends identified, list the major scientific studies that discuss the causes of these changes.",
"Summarize the findings of the listed studies, focusing on the impact of climate change on marine ecosystems.",
"Propose three strategies to mitigate the impact of climate change on marine ecosystems based on the summarized findings."
]

import os
import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# Write a prompt to generate summaries of legal patents using an LLM.

from langchain_openai import OpenAI, ChatOpenAI

model_name = "gpt-4o-mini"
llm = ChatOpenAI(model=model_name)

print(llm.model_name)



def prompt_chain(initial_prompt, follow_up_prompts):
    result = llm.invoke(initial_prompt)
    if result is None:
        return "Initial prompt failed."
    print(f"Initial output: {result.content}\n")
    for i, prompt in enumerate(follow_up_prompts, 1):
        full_prompt = f"{prompt}\n\nPrevious output: {result.content}"
        result = llm.invoke(full_prompt)
        if result is None:
            return f"Prompt {i} failed."
        print(f"Step {i} output: {result.content}\n")
    return result


final_result = prompt_chain(prompts[0], prompts[1:])
print("Final result:", final_result.content )



