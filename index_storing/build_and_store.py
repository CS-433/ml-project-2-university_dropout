from llama_index import VectorStoreIndex
from llama_index import Document
from llama_index import Prompt
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext
from llama_index.text_splitter import SentenceSplitter, TokenTextSplitter
from transformers import AutoTokenizer
from llama_index import set_global_tokenizer
import json
import sys

model_name = sys.argv[1]
knowledge_base_path = sys.argv[2]
persist_dir_path = sys.argv[3]


tokenizer = AutoTokenizer.from_pretrained(model_name)
set_global_tokenizer(tokenizer)

with open(knowledge_base_path, 'r') as f:
    js = json.load(f)
    text_chunks = js

text_splitter = SentenceSplitter(chunk_size=384, chunk_overlap=64)

documents = []
for i, text in enumerate(text_chunks):

    chunks = text_splitter.split_text(text['transcript'])
    for chunk in chunks:
        doc = Document(
            text=chunk+f'\nMedia ID: {text["media_id"]}', 
            doc_id=f"doc_id_{i}"
        )
        documents.append(doc)

embed_model = HuggingFaceEmbedding(
    model_name=model_name
)

service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=service_context)
vector_index.storage_context.persist(persist_dir=f"{persist_dir_path}/{model_name}")