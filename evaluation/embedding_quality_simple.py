from llama_index.readers import SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
import json
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index import Document
from llama_index import Prompt
from llama_index import StorageContext, load_index_from_storage

import torch
from transformers import BitsAndBytesConfig

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import json
from json import JSONDecodeError
from tqdm import tqdm
import sys

model_name = sys.argv[1]
dataset_path = sys.argv[2]
persist_dir_path = sys.argv[3]

template = (
    "Vous faites partie du système de questions-réponses RAG. Vous avez une question et des paragraphes qui s'y rapportent. Chaque paragraphe se termine par la chaîne \"Media ID\". Répondez à la question en utilisant ces paragraphes. Après votre réponse, écrivez \"Media ID\" des paragraphes que vous avez utilisés pour obtenir la réponse dans le champ \"media_id\". \n"
    "Paragraphes pertinents.\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Repond la question: {query_str}.\n"
    "Vous devez répondre à la question au format JSON: {{\"answer\": \"<votre réponse>\", \"media_id\": \"<identifiant de média utilisé>\"}}. Vous devez inclure tous les mots de votre réponse en JSON uniquement, pas d'autres formats.\n"
    "Si vous ne pouvez pas donner de réponse sur la base des informations fournies, vous devez toujours suivre la structure json.\n"
)

embed_model = HuggingFaceEmbedding(
    model_name=model_name
)

service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model)
storage_context = StorageContext.from_defaults(persist_dir=f'{persist_dir_path}/{model_name}')
vector_index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
qa_template = Prompt(template)

query_engine = vector_index.as_query_engine(response_mode="compact", text_qa_template=qa_template, similarity_top_k=4)

with open(dataset_path, 'r') as f:
    dataset = json.load(f)

correct = 0

for example in tqdm(dataset):
    question = example['question']
    transcript = example['transcript']
    response_info = query_engine.query(question)
    texts = [
        (node.node.text, node.score) for node in response_info.source_nodes
    ]
    f = 0
    ftext = texts[0][0]
    for text, score in texts:
        if 'Media ID' in text:
            text = text[: text.find('Media ID')].strip()
        if text.lower() in transcript.lower():
            f = 1
            break
    if f:
        correct += 1

print("RESULT")
print(model_name)
print("CORRECT:", correct)
print("TOTAL:", len(dataset))
print("CORRECT RATIO", correct / len(dataset))