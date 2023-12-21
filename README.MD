# Retrieval Augmented Generation on RTS

This project was developed as part of the EPFL Machine Learning course (2023).

## Authors

- Viacheslav Surkov
- Semen Matrenok
- Daniil Likhobaba

## Summary

This repository contains code for building RAG pipeline and scripts for dataset generation and pipeline evaluation.

## Usage

The code was tested with Python 3.10 and CUDA 12

Install requirements

    pip install -r requirements.txt

Store OpenAI API key to the environment

    os.environ["OPENAI_API_KEY"] = <YOUR API KEY>

### Dataset Generation

    python dataset_generation/script_gpt4.py <path to data> <store path>

Collection of transcripts in ```<path to data>``` should a JSON file in the following format:

    [
        ...
        {
            "transcript": <example transcript>: str,
            "media_id": <example Media ID>: str
        },
        ...
    ]

The dataset of 100 examples will be saved to ```<store path>```.

### Models

First, construct and store embedding index on the dataset.

    python index_storing/build_and_store.py <model name> <path to data> <persist dir path>

```<model name>``` is an embedding model name from huggingface, ```<path to data>``` is a filepath to the dataset (the format is the same as for Dataset Generation section), ```<persist dir path>``` is path to the directory to store index.

To launch the pipeline as service running on port 8000:

    bash models/start_app.sh

It can be configured via ```models/config.json``` configuration file. Refer to it to get the list of configuration parameters.

Querying is performed with HTTP POST requests, for example:

    import requests
    import json
    x = requests.post('http://127.0.0.1:8000/', json={
        'question': question, 
        'techniques': ['cot']})
    print(json.loads(x.text))

```techniques``` is the list of the prompt techniques to apply. Refer to ```models/config.json``` file for the list of implemented techniques.


### Evaluation

To test embeddings:

    python evaluation/embedding_quality_simple.py <model name> <dataset path> <persist dir path>

```<dataset path>``` and ```<persist dir path>``` are paths to the generated dataset and the constructed index.

To test whole pipeline correctness

    python evaluation/quality_metrics.py <dataset path>