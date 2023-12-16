export CUDA_VISIBLE_DEVICES=0
uvicorn models.main:app --reload --reload-include *.json