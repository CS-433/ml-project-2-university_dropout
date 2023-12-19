from fastapi import FastAPI
from baseline import Baseline
from pydantic import BaseModel
from json import JSONDecodeError
import json
from typing import List

app = FastAPI()
class Question(BaseModel):
    question: str

baseline = Baseline()


@app.post("/")
def read_root(question: Question):
    prompt = question.question
    response_info = baseline(prompt)
    
    return response_info.response
