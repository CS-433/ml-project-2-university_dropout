export CUDA_VISIBLE_DEVICES=0
uvicorn main:app --reload --reload-include *.json