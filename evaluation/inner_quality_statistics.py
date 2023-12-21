from langdetect import detect
from collections import defaultdict
from langdetect.lang_detect_exception import LangDetectException
import sys
import json

path_to_file = sys.argv[1]

with open(path_to_file, 'r') as f:
    lines = f.readlines()


groups = defaultdict(lambda: [])

for line in lines:
    answer = json.loads(line)
    if isinstance(answer['model_output'], str):
        groups[tuple(answer['prompt_techniques'])].append(answer['model_output'])

def nonFrenchRatio(answers):
    langs = []
    for answer in answers:
        try:
            langs.append(detect(answer) != 'fr')
        except:
            langs.append(True)
    return sum(langs) / len(langs)

def averageLen(answers):
    lengths = [len(answer.split()) for answer in answers]
    return sum(lengths) / len(lengths)

for techniques, answers in groups.items():
    print(techniques)
    print("Not french:", nonFrenchRatio(answers))
    print("Average length:", averageLen(answers))