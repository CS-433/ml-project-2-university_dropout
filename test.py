import requests
import json

questions = [
    "Combien de buts Merci a-t-il marqué dans le MSN Trident ?"
]

techniques = ['cot']

for question in questions:
    x = requests.post('http://127.0.0.1:8000/', json={'question': question, 'techniques': techniques})
    print('Question:')
    print(question)
    print('Answer:')
    print(json.loads(x.text)['answer']['answer'])
    print(json.loads(x.text)['answer']['media_id'])
    print()
