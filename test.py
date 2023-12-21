import requests
import json

questions = [
    "Combien de villas à 1 million d'euros pouvons-nous acheter avec 3000 milliards d'euros?",
    "Combien de villas à 1 million d'euros pouvons-nous acheter avec 3000 milliards d'euros?",
    "Combien de villas à 1 million d'euros pouvons-nous acheter avec 3000 milliards d'euros?"
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
