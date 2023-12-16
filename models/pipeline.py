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
from pydantic import BaseModel
import json
from json import JSONDecodeError
from models.utils import make_prompt

class RAGPipeline:
    def __init__(self, config):
        with open(config) as f:
            self.config = json.load(f)

        self.load_models()

    def load_models(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        llm = HuggingFaceLLM(
            model_name=self.config['model'],
            tokenizer_name=self.config['model'],
            query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
            
            context_window=3900,
            max_new_tokens=1024,
            model_kwargs={"quantization_config": quantization_config},
            # tokenizer_kwargs={},
            generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            device_map="auto",
        ) 

        template = make_prompt(techniques=self.config['prompt_techniques'])
        qa_template = Prompt(template)

        embed_model = HuggingFaceEmbedding(
            model_name=self.config['embeddings']
        )

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        storage_context = StorageContext.from_defaults(persist_dir=self.config['persist_dir']+self.config['embeddings'])
        vector_index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
        self.query_engine = vector_index.as_query_engine(response_mode="compact", text_qa_template=qa_template)

    def change_prompt(self, prompt_techniques):
        template = make_prompt(techniques=prompt_techniques)
        qa_template = Prompt(template)
        self.query_engine.text_qa_template = qa_template

    def __call__(self, question: str, force_prompt_techniques=None):
        if force_prompt_techniques is not None:
            self.change_prompt(force_prompt_techniques)

        response_info = self.query_engine.query(question)
        llm_response = response_info 
        print(llm_response)

        if force_prompt_techniques is not None:
            self.change_prompt(self.config['prompt_techniques'])

        return llm_response

