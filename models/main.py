from fastapi import FastAPI
from models.pipeline import RAGPipeline
from pydantic import BaseModel
from json import JSONDecodeError
import json

app = FastAPI()
class Question(BaseModel):
    question: str

def extract_json_substr(response):
    start = response.find('{')
    finish = response.rfind('}')
    if start == -1 or finish == -1:
        raise JSONDecodeError
    return response[start:finish+1]

rag_pipeline = RAGPipeline(config='models/config.json')

@app.post("/")
def read_root(question: Question):
    for _ in range(5):
        try:
            prompt = question.question
            response_info = rag_pipeline(prompt)
            llm_response = extract_json_substr(response_info.response)
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