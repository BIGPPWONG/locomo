#!/bin/bash

# sets necessary environment variables
source scripts/env.sh

# LangGraph/Assistant API with different configurations
# Example 1: Mock mode (for testing)
# python3 task_eval/evaluate_qa.py \
#     --data-file $DATA_FILE_PATH --out-file $OUT_DIR/langgraph_mock_$QA_OUTPUT_FILE \
#     --model langgraph-mock --langgraph-api-type mock

# Example 2: Custom endpoint (with concurrent control & model specification)
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/langgraph_custom_$QA_OUTPUT_FILE \
    --model langgraph-custom --langgraph-api-type custom_endpoint \
    --langgraph-endpoint "http://127.0.0.1:8001" \
    --langgraph-api-key "your-api-key" \
    --langgraph-max-concurrent 2 \
    --langgraph-ingest-model "iflow/glm-4.5" \
    --langgraph-retrieve-model "nvidia/qwen3-next-80b-a3b-instruct" \
    # --skip-langgraph-rag # Skip RAG ingest for faster testing

# Example 2b: Custom endpoint with RAG ingestion (using ingest model)
# python3 task_eval/evaluate_qa.py \
#     --data-file $DATA_FILE_PATH --out-file $OUT_DIR/langgraph_custom_rag_$QA_OUTPUT_FILE \
#     --model langgraph-custom --langgraph-api-type custom_endpoint \
#     --langgraph-endpoint "http://127.0.0.1:8001" \
#     --langgraph-api-key "your-api-key" \
#     --langgraph-max-concurrent 5 \
#     --langgraph-ingest-model "text-embedding-ada-002" \
#     --langgraph-retrieve-model "nvidia/qwen3-next-80b-a3b-instruct"

# Example 3: OpenAI Assistant API (with concurrent control)
# python3 task_eval/evaluate_qa.py \
#     --data-file $DATA_FILE_PATH --out-file $OUT_DIR/openai_assistant_$QA_OUTPUT_FILE \
#     --model openai-assistant --langraph-api-type openai_assistant \
#     --langgraph-api-key "your-openai-key" \
#     --langgraph-max-concurrent 10

# Example 4: LangGraph Cloud with custom concurrency
# python3 task_eval/evaluate_qa.py \
#     --data-file $DATA_FILE_PATH --out-file $OUT_DIR/langgraph_cloud_$QA_OUTPUT_FILE \
#     --model langgraph-cloud --langgraph-api-type langgraph_cloud \
#     --langgraph-api-key "your-langgraph-key" \
#     --langgraph-max-concurrent 15

echo "LangGraph evaluation completed!"