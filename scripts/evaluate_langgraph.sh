#!/bin/bash

# sets necessary environment variables
source scripts/env.sh

# LangGraph/Assistant API with different configurations
# Example 1: Mock mode (for testing)
# python3 task_eval/evaluate_qa.py \
#     --data-file $DATA_FILE_PATH --out-file $OUT_DIR/langgraph_mock_$QA_OUTPUT_FILE \
#     --model langgraph-mock --langgraph-api-type mock

# Example 2: Custom endpoint (with concurrent control & model specification)
# python3 task_eval/evaluate_qa.py \
#     --data-file $DATA_FILE_PATH --out-file $OUT_DIR/langgraph_custom_$QA_OUTPUT_FILE \
#     --model langgraph-custom --langgraph-api-type custom_endpoint \
#     --langgraph-endpoint "http://127.0.0.1:8001" \
#     --langgraph-api-key "your-api-key" \
#     --langgraph-max-concurrent 10 \
#     --langgraph-ingest-model "qwen3-coder-30b-a3b-instruct" \
#     --langgraph-retrieve-model "qwen3-coder-30b-a3b-instruct" \
#     --skip-langgraph-retrieve

# Example 2a: Skip retrieve phase entirely (fastest testing)
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



# Example 2b: Custom endpoint with RAG ingestion (using ingest model)
# python3 task_eval/evaluate_qa.py \
#     --data-file $DATA_FILE_PATH --out-file $OUT_DIR/langgraph_custom_rag_$QA_OUTPUT_FILE \
#     --model langgraph-custom --langgraph-api-type custom_endpoint \
#     --langgraph-endpoint "http://127.0.0.1:8001" \
#     --langgraph-api-key "your-api-key" \
#     --langgraph-max-concurrent 5 \
#     --langgraph-ingest-model "text-embedding-ada-002" \
#     --langgraph-retrieve-model "nvidia/qwen3-next-80b-a3b-instruct" \
#     --categories "5"  # Only test category 5 questions

# Example 3: OpenAI Assistant API (with concurrent control)
# python3 task_eval/evaluate_qa.py \
#     --data-file $DATA_FILE_PATH --out-file $OUT_DIR/openai_assistant_$QA_OUTPUT_FILE \
#     --model openai-assistant --langraph-api-type openai_assistant \
#     --langgraph-api-key "your-openai-key" \
#     --langgraph-max-concurrent 10 \
#     --categories "3,4,5" # Test categories 3, 4, and 5

# Example 4: LangGraph Cloud with custom concurrency
# python3 task_eval/evaluate_qa.py \
#     --data-file $DATA_FILE_PATH --out-file $OUT_DIR/langgraph_cloud_$QA_OUTPUT_FILE \
#     --model langgraph-cloud --langgraph-api-type langgraph_cloud \
#     --langgraph-api-key "your-langgraph-key" \
#     --langgraph-max-concurrent 15 \
#     --categories "1,3,5" # Quickly debug with specific categories

echo "LangGraph evaluation completed!"