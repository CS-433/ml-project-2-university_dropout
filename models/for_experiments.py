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




with open("config.json") as f:
    config = json.load(f)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    
    context_window=3900,
    max_new_tokens=1024,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    device_map="auto",
)

with open('../../data/knowledge_base_100.json', 'r') as f:
    text_chunks = json.load(f)

documents = []
for i, text in enumerate(text_chunks):
    doc = Document(text=text, doc_id=f"doc_id_{i}")
    documents.append(doc)


template = make_prompt(techniques=config['prompt_techniques'])
qa_template = Prompt(template)

embed_model = HuggingFaceEmbedding(
    model_name="dangvantuan/sentence-camembert-large"
)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
# vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
storage_context = StorageContext.from_defaults(persist_dir='/home/ragteam/experiments/index_storage/dangvantuan/sentence-camembert-large_example')
vector_index = load_index_from_storage(storage_context=storage_context, service_context=service_context)

query_engine = vector_index.as_query_engine(response_mode="compact", text_qa_template=qa_template)

app = FastAPI()
class Question(BaseModel):
    question: str

def extract_json_substr(response):
    start = response.find('{')
    finish = response.rfind('}')
    if start == -1 or finish == -1:
        raise JSONDecodeError
    return response[start:finish+1]

@app.post("/")
def read_root(question: Question):
    for _ in range(5):
        try:
            prompt = question.question
            response_info = query_engine.query(prompt)
            llm_response = response_info.response
            print(llm_response)
            llm_response = extract_json_substr(llm_response)
            print(llm_response)
            texts = [
                (node.node.text, node.score) for node in response_info.source_nodes
            ]
            parsed_json = json.loads(llm_response)
            return {
                'answer': parsed_json,
                'texts': texts
            }
        except JSONDecodeError:
            continue

