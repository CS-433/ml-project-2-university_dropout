from llama_index.llms import HuggingFaceLLM
from transformers import BitsAndBytesConfig
import torch

class Baseline:
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )

        self.llm = HuggingFaceLLM(
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
            
            context_window=3900,
            max_new_tokens=1024,
            model_kwargs={"quantization_config": quantization_config},
            # tokenizer_kwargs={},
            generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            device_map="auto",
        )

    def __call__(self, question):
        return self.llm.complete(question)
