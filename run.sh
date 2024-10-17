#!/bin/bash

# Setup for running training models for four different cities

# Ensure CUDA devices are correctly assigned (assuming a multi-GPU setup)
export CUDA_VISIBLE_DEVICES=0

# Define common parameters
EPOCHS=30
BASE_LR=0.0003
MODEL_NAME="MobilityBERTMoE"
MODEL_PATH="path_to_pretrained_model.pth"
LOCATION_EMBEDDING_LR=0.0001  # This can be adjusted or removed if not needed

# Training for City A
CITY="A"
NUM_LOCATION_IDS=40000
HIDDEN_SIZE=256
HIDDEN_LAYERS=24
ATTENTION_HEADS=16
DAY_EMBED_SIZE=64
TIME_EMBED_SIZE=64
DAY_OF_WEEK_EMBED_SIZE=64
WEEKDAY_EMBED_SIZE=32
LOCATION_EMBED_SIZE=256
DROPOUT=0.2
MAX_SEQ_LENGTH=3648  # Example length, adjust as needed

python main.py --model_name $MODEL_NAME \
               --num_location_ids $NUM_LOCATION_IDS \
               --hidden_size $HIDDEN_SIZE \
               --hidden_layers $HIDDEN_LAYERS \
               --attention_heads $ATTENTION_HEADS \
               --day_embedding_size $DAY_EMBED_SIZE \
               --time_embedding_size $TIME_EMBED_SIZE \
               --day_of_week_embedding_size $DAY_OF_WEEK_EMBED_SIZE \
               --weekday_embedding_size $WEEKDAY_EMBED_SIZE \
               --location_embedding_size $LOCATION_EMBED_SIZE \
               --dropout $DROPOUT \
               --max_seq_length $MAX_SEQ_LENGTH \
               --lr $BASE_LR \
               --location_embedding_lr $LOCATION_EMBEDDING_LR \
               --num_epochs $EPOCHS \
               --device "cuda:0"


# Fine-tune for City B, C, D
CITY="B"
NUM_LOCATION_IDS=40000
HIDDEN_SIZE=256
HIDDEN_LAYERS=24
ATTENTION_HEADS=16
DAY_EMBED_SIZE=64
TIME_EMBED_SIZE=64
DAY_OF_WEEK_EMBED_SIZE=64
WEEKDAY_EMBED_SIZE=32
LOCATION_EMBED_SIZE=256
DROPOUT=0.2
MAX_SEQ_LENGTH=3648  # Example length, adjust as needed

python main.py --model_name $MODEL_NAME \
               --num_location_ids $NUM_LOCATION_IDS \
               --hidden_size $HIDDEN_SIZE \
               --hidden_layers $HIDDEN_LAYERS \
               --attention_heads $ATTENTION_HEADS \
               --day_embedding_size $DAY_EMBED_SIZE \
               --time_embedding_size $TIME_EMBED_SIZE \
               --day_of_week_embedding_size $DAY_OF_WEEK_EMBED_SIZE \
               --weekday_embedding_size $WEEKDAY_EMBED_SIZE \
               --location_embedding_size $LOCATION_EMBED_SIZE \
               --dropout $DROPOUT \
               --max_seq_length $MAX_SEQ_LENGTH \
               --lr $BASE_LR \
               --location_embedding_lr $LOCATION_EMBEDDING_LR \
               --num_epochs $EPOCHS \
               --device "cuda:0" \
               --model_path $MODEL_PATH
