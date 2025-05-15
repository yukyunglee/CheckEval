# #!/bin/bash
MODEL_TYPE="Mistral-Small-Instruct-2409"
MODEL=.Models/Mistral-Small-Instruct-2409
BASE_URL=<ADDYOURBASEURL>

# summeval
TEMPLATE_TYPE="summeval"
ASPECTS="coherence consistency fluency relevance"

DATA_PATH=./data/${TEMPLATE_TYPE}/${TEMPLATE_TYPE}_result.csv
SAVE_DIR=./etc/checkeval_results/${MODEL_TYPE}/${TEMPLATE_TYPE}/multi
QUESTION_VERSIONS=("seed" "diversification" "elaboration")

for QUESTION_VERSION in "${QUESTION_VERSIONS[@]}"; do
    python inference_checkeval.py \
    --data_path $DATA_PATH \
    --base_url $BASE_URL \
    --model $MODEL \
    --save_dir $SAVE_DIR \
    --question_version $QUESTION_VERSION \
    --aspects $ASPECTS \
    --template_type $TEMPLATE_TYPE
done

# topical_chat
TEMPLATE_TYPE="topical_chat"
ASPECTS="coherence engagingness groundedness naturalness"

DATA_PATH=./data/${TEMPLATE_TYPE}/${TEMPLATE_TYPE}_result.csv
SAVE_DIR=./etc/checkeval_results/${MODEL_TYPE}/${TEMPLATE_TYPE}/multi
QUESTION_VERSIONS=("seed" "diversification" "elaboration")

for QUESTION_VERSION in "${QUESTION_VERSIONS[@]}"; do
    python inference_checkeval.py \
    --data_path $DATA_PATH \
    --base_url $BASE_URL \
    --model $MODEL \
    --save_dir $SAVE_DIR \
    --question_version $QUESTION_VERSION \
    --aspects $ASPECTS \
    --template_type $TEMPLATE_TYPE
done


# topical_chat
TEMPLATE_TYPE="topical_chat"
python inference_geval.py \
--base_url $BASE_URL \
--model $MODEL \
--save_dir ./etc/checkeval_results/${MODEL_TYPE}/${TEMPLATE_TYPE}/geval

# summeval
TEMPLATE_TYPE="summeval"
python inference_geval.py \
--base_url $BASE_URL \
--model $MODEL \
--save_dir ./etc/checkeval_results/${MODEL_TYPE}/${TEMPLATE_TYPE}/geval