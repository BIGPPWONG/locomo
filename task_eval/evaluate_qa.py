import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, json
from tqdm import tqdm
import argparse
from global_methods import set_openai_key, set_anthropic_key, set_gemini_key
from task_eval.evaluation import eval_question_answering, eval_llm_judge_qa
from task_eval.evaluation_stats import analyze_aggr_acc
from task_eval.gpt_utils import get_gpt_answers
from task_eval.claude_utils import get_claude_answers
from task_eval.gemini_utils import get_gemini_answers
from task_eval.hf_llm_utils import init_hf_model, get_hf_answers
from task_eval.langgraph_utils import get_langgraph_answers, create_langgraph_config

import numpy as np
import google.generativeai as genai

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--use-4bit', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="contriever")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--categories', type=str, default="",
                        help="Comma-separated list of categories to test (e.g., '1,2,3'). If empty, test all categories")
    parser.add_argument('--use-llm-judge', action="store_true",
                        help="Use LLM as judge for evaluation instead of traditional metrics")
    parser.add_argument('--judge-model', type=str, default="gpt-4o",
                        help="OpenAI model to use as LLM judge (default: 'gpt-4o')")
    parser.add_argument('--judge-max-concurrent', type=int, default=5,
                        help="Maximum concurrent LLM judge calls (default: 5)")

    # LangGraph/Assistant API specific arguments
    parser.add_argument('--langgraph-api-type', type=str, default="mock",
                        help="API type: mock, langgraph_cloud, openai_assistant, custom_endpoint")
    parser.add_argument('--langgraph-endpoint', type=str, default="",
                        help="Custom endpoint URL for LangGraph")
    parser.add_argument('--langgraph-api-key', type=str, default="",
                        help="API key for LangGraph service")
    parser.add_argument('--langgraph-max-concurrent', type=int, default=5,
                        help="Maximum concurrent writes to LangGraph store (default: 5)")
    parser.add_argument('--skip-langgraph-rag', action="store_true",
                        help="Skip RAG context injection for LangGraph (useful when context is already stored)")
    parser.add_argument('--skip-langgraph-retrieve', action="store_true",
                        help="Skip retrieve phase for LangGraph (useful when testing without making API calls)")
    parser.add_argument('--langgraph-ingest-model', type=str, default="nvidia/qwen3-next-80b-a3b-instruct",
                        help="Model name for RAG context ingestion in LangGraph (e.g., 'text-embedding-ada-002')")
    parser.add_argument('--langgraph-retrieve-model', type=str, default="nvidia/qwen3-next-80b-a3b-instruct",
                        help="Model name for answer retrieval in LangGraph (e.g., 'gpt-4')")

    args = parser.parse_args()
    return args


def main():

    # get arguments
    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    if 'gpt' in args.model:
        # set openai API key
        set_openai_key()

    elif 'claude' in args.model:
        # set openai API key
        set_anthropic_key()

    elif 'gemini' in args.model:
        # set openai API key
        set_gemini_key()
        if args.model == "gemini-pro-1.0":
            model_name = "models/gemini-1.0-pro-latest"

        gemini_model = genai.GenerativeModel(model_name)

    elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
        hf_pipeline, hf_model_name = init_hf_model(args)

    elif 'langgraph' in args.model or 'assistant' in args.model:
        # Initialize LangGraph/Assistant API config
        langgraph_config = create_langgraph_config(
            api_type=args.langgraph_api_type,
            endpoint=args.langgraph_endpoint,
            api_key=args.langgraph_api_key,
            max_concurrent_writes=args.langgraph_max_concurrent,
            ingest_model=args.langgraph_ingest_model,
            retrieve_model=args.langgraph_retrieve_model,
            skip_retrieve=args.skip_langgraph_retrieve
        )

    else:
        raise NotImplementedError


    # load conversations
    samples = json.load(open(args.data_file))
    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)
    model_key = "%s" % args.model if not args.use_rag else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k)
    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}


    for data in tqdm(samples):

        out_data = {'sample_id': data['sample_id']}
        if data['sample_id'] in out_samples:
            out_data['qa'] = out_samples[data['sample_id']]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()

        # Filter QA pairs by category if specified
        if args.categories:
            target_categories = [int(c.strip()) for c in args.categories.split(',')]
            filtered_qa = []
            for qa in out_data['qa']:
                if 'category' in qa and qa['category'] in target_categories:
                    filtered_qa.append(qa)
            if not filtered_qa:
                print(f"Skipping sample {data['sample_id']}: no QA pairs match categories {target_categories}")
                continue
            # 过滤原数据和输出数据中的qa
            out_data['qa'] = filtered_qa
            data['qa'] = filtered_qa

        if 'gpt' in args.model:
            # get answers for each sample
            answers = get_gpt_answers(data, out_data, prediction_key, args)
        elif 'claude' in args.model:
            answers = get_claude_answers(data, out_data, prediction_key, args)
        elif 'gemini' in args.model:
            answers = get_gemini_answers(gemini_model, data, out_data, prediction_key, args)
        elif any([model_name in args.model for model_name in ['gemma', 'llama', 'mistral']]):
            answers = get_hf_answers(data, out_data, args, hf_pipeline, hf_model_name)
        elif 'langgraph' in args.model or 'assistant' in args.model:
            answers = get_langgraph_answers(data, out_data, prediction_key, args, langgraph_config)
        else:
            raise NotImplementedError

        # evaluate individual QA samples and save the score
        if args.use_llm_judge:
            # Use LLM as judge for evaluation
            exact_matches, lengths, recall = eval_llm_judge_qa(answers['qa'], prediction_key, args.judge_model, args.judge_max_concurrent)
        else:
            # Use traditional evaluation metrics
            exact_matches, lengths, recall = eval_question_answering(answers['qa'], prediction_key)

        for i in range(0, len(answers['qa'])):
            answers['qa'][i][model_key + '_f1'] = round(exact_matches[i], 3)
            if args.use_rag and len(recall) > 0:
                answers['qa'][i][model_key + '_recall'] = round(recall[i], 3)

        out_samples[data['sample_id']] = answers

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(list(out_samples.values()), f, indent=2)

    
    analyze_aggr_acc(args.data_file, args.out_file, args.out_file.replace('.json', '_stats.json'),
                model_key, model_key + '_f1', rag=args.use_rag)
    # encoder=tiktoken.encoding_for_model(args.model))


main()

