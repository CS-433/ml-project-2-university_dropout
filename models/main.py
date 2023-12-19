from fastapi import FastAPI
from pipeline import RAGPipeline
from pydantic import BaseModel
from json import JSONDecodeError
import json
from typing import List

app = FastAPI()
class Question(BaseModel):
    question: str
    techniques: List[str]

def extract_json_substr(response):
    start = response.find('{')
    finish = response.find('}')
    if start == -1 or finish == -1:
        raise JSONDecodeError
    return response[start:finish+1]

def hotfix(llm_response, pos):
    list_of_chars = list(llm_response)
    list_of_chars[pos] = ','
    return ''.join(list_of_chars)

def parse_json(response_info):
    llm_response = extract_json_substr(response_info.response)
    try:
        parsed_json = json.loads(llm_response)
    except JSONDecodeError as e:
        llm_response = hotfix(llm_response, e.pos)
        parsed_json = json.loads(llm_response)
    return parsed_json


rag_pipeline = RAGPipeline(config='config.json')


def try_generate(question: Question):
    prompt = question.question
    techniques = question.techniques
    response_info = rag_pipeline(prompt, techniques)
    
    texts = [
        (node.node.text, node.score) for node in response_info.source_nodes
    ]
    try:
        parsed_json = parse_json(response_info)
    except JSONDecodeError as e:
        return False        

    return {
        'answer': parsed_json,
        'texts': texts
    }


@app.post("/")
def read_root(question: Question):
    for _ in range(5):
        response = try_generate(question)
        if response != False:
            return response
    return {
        'answer': {
            "answer": "Je n'ai pas réussi à trouver la réponse.",
            "media_id": ""
        },
        'texts': []
    }