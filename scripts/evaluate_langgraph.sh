#!/bin/bash

# sets necessary environment variables
source scripts/env.sh

# Step 1: ingest data into LangGraph
echo "Ingesting data into LangGraph..."
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/langgraph_custom_$QA_OUTPUT_FILE \
    --model langgraph-custom --langgraph-api-type custom_endpoint \
    --langgraph-endpoint "http://127.0.0.1:8001" \
    --langgraph-api-key "your-api-key" \
    --langgraph-max-concurrent 10 \
    --langgraph-ingest-model "qwen3-coder-30b-a3b-instruct" \
    --langgraph-retrieve-model "qwen3-coder-30b-a3b-instruct" \
    --skip-langgraph-retrieve

# Clean up Step 1 output files
echo "Cleaning up Step 1 output files..."
rm -f $OUT_DIR/langgraph_custom*.json

# Step 2: evaluate QA using LangGraph with RAG
echo "Evaluating QA using LangGraph with RAG..."
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/langgraph_custom_$QA_OUTPUT_FILE \
    --model langgraph-custom --langgraph-api-type custom_endpoint \
    --langgraph-endpoint "http://127.0.0.1:8001" \
    --langgraph-api-key "your-api-key" \
    --langgraph-max-concurrent 10 \
    --langgraph-ingest-model "4090/qwen3-coder-30b-a3b-instruct" \
    --langgraph-retrieve-model "h800/glm-4.5" \
    --skip-langgraph-rag \
    --use-llm-judge \
    --categories "1,2,3,4,5" \
    --judge-model "4090/qwen3-coder-30b-a3b-instruct" \
    --judge-max-concurrent 10 

echo "LangGraph evaluation completed!"