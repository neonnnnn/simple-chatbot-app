#!/bin/bash
echo "###### installing dependencies ######"
pip3 install -r requirements.txt

echo "###### downloading the model ######"
python3 ./download_model.py

echo "###### archiving the model ######"
torch-model-archiver \
  --model-name chatbot \
  --version 1.0 \
  --serialized-file download/pytorch_model.bin \
  --handler ./handler.py \
  --extra-files "./download/config.json,./download/generation_config.json,./download/merges.txt,./download/special_tokens_map.json,./download/tokenizer_config.json,./download/tokenizer.json,./download/vocab.json,./setup_config.json"

mkdir model_store
mv chatbot.mar model_store/