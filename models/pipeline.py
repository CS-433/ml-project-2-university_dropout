from llama_index.readers import SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index import Document
from llama_index import Prompt
from llama_index import StorageContext, load_index_from_storage
from llama_index.llms import LlamaCPP

import torch
from transformers import BitsAndBytesConfig
from typing import Union
from pydantic import BaseModel
import json
from json import JSONDecodeError
from utils import make_prompt

class RAGPipeline:
    def __init__(self, config):
        with open(config) as f:
            self.config = json.load(f)

        self.load_models()

    def load_huggingface_llm(self):

        generate_kwargs=self.config['generate_kwargs']

        llm = HuggingFaceLLM(
            model_name=self.config['model'],
            tokenizer_name=self.config['model'],
            context_window=3900,
            max_new_tokens=1024,
            query_wrapper_prompt=PromptTemplate("<|système|>\n</s>\n<|usager|>\n{query_str}</s>\n<|assistant|>\n"),
    
            model_kwargs={"torch_dtype": torch.float16},
            generate_kwargs=generate_kwargs,
            device_map="auto",
        )
        return llm

    def load_models(self):
        llm = self.load_huggingface_llm()

        template = make_prompt(techniques=self.config['prompt_techniques'])
        qa_template = Prompt(template)

        embed_model = HuggingFaceEmbedding(
            model_name=self.config['embeddings']
        )

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        storage_context = StorageContext.from_defaults(persist_dir=f"{self.config['persist_dir']}/{self.config['embeddings']}")
        self.vector_index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
        self.query_engine = self.vector_index.as_query_engine(response_mode="compact", text_qa_template=qa_template, similarity_top_k=4)

    def change_prompt(self, prompt_techniques):
        template = make_prompt(techniques=prompt_techniques)
        qa_template = Prompt(template)
        self.query_engine = self.vector_index.as_query_engine(response_mode="compact", text_qa_template=qa_template, similarity_top_k=4)

    def __call__(self, question: str, force_prompt_techniques=None):
        if force_prompt_techniques is not None:
            self.change_prompt(force_prompt_techniques)

        response_info = self.query_engine.query(question)
        llm_response = response_info 
        
        if force_prompt_techniques is not None:
            self.change_prompt(self.config['prompt_techniques'])

        return llm_response

