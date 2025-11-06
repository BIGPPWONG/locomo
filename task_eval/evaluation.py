import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
import os
import asyncio
from pathlib import Path
from bert_score import score
from nltk.stem import PorterStemmer
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from tqdm import tqdm
from diskcache import Cache
ps = PorterStemmer()

LENGTH_THRESHOLD = 5

# Cache directory for LLM judge results
LLM_JUDGE_CACHE_DIR = (Path(__file__).resolve().parent / ".." / "cache" / "llm_judge").resolve()
LLM_JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LLM_JUDGE_CACHE = Cache(str(LLM_JUDGE_CACHE_DIR))


def make_llm_judge_cache_key(model: str, question, answer, prediction, **metadata) -> str:
    """
    Create a stable cache key for judging results. Values are serialized to JSON
    to handle lists/dicts while remaining ASCII-friendly.
    """
    payload = {
        "model": model,
        "question": question,
        "answer": answer,
        "prediction": prediction,
    }
    if metadata:
        payload.update(metadata)
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):

    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):

    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    # print('# EM #', prediction, ' | ', ground_truth, ' #', set(prediction.split()) == set(ground_truth.split()))
    # return normalize_answer(prediction) == normalize_answer(ground_truth)
    return set(prediction.split()) == set(ground_truth.split())
    
# def bert_score(prediction, ground_truths):
#     prediction = normalize_answer(prediction)
#     values = []
#     for ground_truth in ground_truths:
#         ground_truth = normalize_answer(ground_truth)
#         P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
#         values.append(R[0].item())
#     print('# BERT # ', normalize_answer(prediction), ' | ', normalize_answer(ground_truth), ' #', P, R, F1)
#     return max(0, max(values))


def bert_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
    # print('# BERT # ', normalize_answer(prediction), ' | ', normalize_answer(ground_truth), ' #', P, R, F1)
    return max(0, F1[0].item())


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # print('# F1 #', prediction, ' | ', ground_truth, ' #', precision, recall, f1)
    # return recall
    return f1


def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    # print('# F1 [multi-answer]#', predictions, ' | ', ground_truths, ' #', np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths]))
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    from rouge import Rouge
    rouge = Rouge()
    prediction = ' '.join([ps.stem(w) for w in normalize_answer(prediction).split()])
    ground_truth = ' '.join([ps.stem(w) for w in normalize_answer(ground_truth).split()])
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]


def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])


## file-level evaluation ... ### 
def eval_recall(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens


def eval_question_answering(qas, eval_key='prediction', metric='f1'):


    all_ems = []
    all_recall = []
    exact_match_count = 0
    f1_count = 0
    answer_lengths = []
    for i, line in enumerate(qas):
        # line = json.loads(line)
        if type(line[eval_key]) == list:
            answer = line['answer']
        else:
            answer = str(line['answer']) if line['category'] != 5 else 'Not mentioned in the conversation'
        if line['category'] == 3:
            answer = answer.split(';')[0].strip()
        
        output = line[eval_key]
        
        # single-hop, temporal, open-domain eval without splitting for sub-answers 
        if line['category'] in [2, 3, 4]:
            all_ems.append(f1_score(output, answer))
        
        # multi-hop eval by splitting entire phrase into sub-answers and computing partial F1 for each
        elif line['category'] in [1]:
            all_ems.append(f1(output, answer))

        # adversarial eval --> check for selection of correct option
        elif line['category'] in [5]:
            if 'no information available' in output.lower() or 'not mentioned' in output.lower():
                all_ems.append(1)
            else:
                all_ems.append(0)
        else:
            print(line)
            raise ValueError
        
        assert i+1 == len(all_ems), all_ems

        if eval_key + '_context' in line and len(line['evidence']) > 0:
            # recall_acc for dialog
            if line[eval_key + '_context'][0].startswith('S'):
                sessions = [e[1:] for e in line[eval_key + '_context']]
                recall_acc = float(sum([ev.split(':')[0][1:] in sessions for ev in line["evidence"]]))/len(line['evidence'])
            else:
                recall_acc = float(sum([ev in line[eval_key + '_context'] for ev in line["evidence"]]))/len(line['evidence'])
            all_recall.append(recall_acc)
        else:
            all_recall.append(1)

    print("{} QA samples evaluated; {} accuracy values".format(len(qas), len(all_ems)))
    lens = 0.0
    return all_ems, lens, all_recall


def eval_fact_checking(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        if answer == ["refutes"]:
            answer = ["refutes", "no", "false"]
        if answer == ["supports"]:
            answer = ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens


def eval_dialogue_system(infile):

    lines = open(infile, 'r').readlines()[1:]

    f1_scores = []
    rl_scores = []
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)
    lens = round(np.mean(answer_lengths), 4)

    return F1, RL, lens


## LLM-as-Judge Evaluation Function (Structured Output) ###


class JudgeScore(BaseModel):
    """Structured output for LLM judge score"""
    reasoning: str = Field(description="Brief reasoning for the judgment (1-2 sentences)")
    accuracy: Literal["correct", "partial", "incorrect"] = Field(description="Accuracy judgment: 'correct' if prediction is accurate, 'partial' if partially correct, 'incorrect' if prediction is wrong")


def eval_llm_judge_qa(qas, eval_key='prediction', judge_model="gpt-4o", max_concurrent=5):
    """
    Evaluate QA pairs using LLM as judge with ChatOpenAI structured output.
    Returns 1.0 for correct answers, 0.5 for partial answers, 0.0 for incorrect answers.

    Args:
        qas: List of QA dictionaries with predictions
        eval_key: Key containing the prediction to evaluate
        judge_model: OpenAI model to use as judge
        max_concurrent: Maximum number of concurrent LLM judge calls (default: 5)

    Returns:
        all_scores: List of scores (1.0 for correct, 0.5 for partial, 0.0 for incorrect)
        lens: Lengths (kept for compatibility)
        all_recall: Recall scores (kept for compatibility)
    """
    import asyncio

    # Run the async version internally for compatibility
    return asyncio.run(eval_llm_judge_qa_async(qas, eval_key, judge_model, max_concurrent))


async def eval_llm_judge_qa_async(qas, eval_key='prediction', judge_model="gpt-4o", max_concurrent=5):
    """
    Async version of eval_llm_judge_qa with concurrent processing.
    """
    # Initialize ChatOpenAI judge with structured output
    llm_judge = ChatOpenAI(
        model=judge_model,
        temperature=0.0,  # Deterministic judging
        # max_tokens=100,
        # request_timeout=30
    ).with_structured_output(JudgeScore, method="json_schema")

    all_scores = []
    all_recall = []
    answer_lengths = []

    # Judge system prompt for evaluation
    judge_system_prompt = """You are an expert evaluator for question-answering systems.
Your task is to determine if a predicted answer is accurate compared to the ground truth answer.

You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.


Judge 'correct' if the predicted answer captures the essential meaning of the ground truth.
Judge 'partial' if the predicted answer contains some correct information but misses key parts or has significant inaccuracies. Examples:
  - Gets the main idea right but misses important details
  - Has some factual errors mixed with correct information
  - Provides related but not directly relevant information
Judge 'incorrect' if the predicted answer is factually wrong or misses key information. Examples:
  - Completely wrong or irrelevant
  - Misses all key information
  - Contradicts the ground truth
  - Provides no useful information"""

    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(max_concurrent)

    # Process all questions concurrently with progress tracking
    total_questions = len(qas)
    with tqdm(total=total_questions, desc=f"LLM Judge evaluation (max {max_concurrent} concurrent)") as pbar:

        async def process_single_question(i, line):
            async with sem:
                try:
                    # Extract ground truth answer and prediction
                    if type(line[eval_key]) == list:
                        answer = line['answer']
                    else:
                        answer = str(line['answer']) if line['category'] != 5 else 'Not mentioned in the conversation'

                    if line['category'] == 3:
                        answer = answer.split(';')[0].strip()

                    prediction = line[eval_key]
                    question = line.get('question', 'Question not available')

                    cache_key = make_llm_judge_cache_key(
                        judge_model,
                        question,
                        answer,
                        prediction,
                        category=line.get('category'),
                        sample_id=line.get('sample_id')
                    )

                    cached_result = LLM_JUDGE_CACHE.get(cache_key)
                    from_cache = cached_result is not None

                    if cached_result:
                        accuracy_score = cached_result.get('score', 0.0)
                        reasoning = cached_result.get('reasoning', '')
                        accuracy_label = cached_result.get('label', 'unknown')
                    else:
                        # Prepare evaluation prompt for LLM judging
                        evaluation_prompt = f"""Question: {question}
Ground Truth Answer: {answer}
Predicted Answer: {prediction}

Is the predicted answer accurate compared to the ground truth?"""

                        messages = [
                            ("system", judge_system_prompt),
                            ("human", evaluation_prompt)
                        ]

                        structured_response: JudgeScore = llm_judge.invoke(messages, temperature=0.0)
                        accuracy_label = structured_response.accuracy

                        # Convert "correct"/"partial"/"incorrect" to 1.0/0.5/0.0
                        if structured_response.accuracy.lower() == "correct":
                            accuracy_score = 1.0
                        elif structured_response.accuracy.lower() == "partial":
                            accuracy_score = 0.5
                        else:
                            accuracy_score = 0.0
                        reasoning = structured_response.reasoning

                        LLM_JUDGE_CACHE[cache_key] = {
                            "score": accuracy_score,
                            "reasoning": reasoning,
                            "label": accuracy_label,
                        }

                    # For compatibility with existing evaluation pipeline
                    recall_score = 1.0
                    if eval_key + '_context' in line and len(line.get('evidence', [])) > 0:
                        # recall_acc for dialog
                        if line[eval_key + '_context'][0].startswith('S'):
                            sessions = [e[1:] for e in line[eval_key + '_context']]
                            recall_score = float(sum([ev.split(':')[0][1:] in sessions for ev in line["evidence"]]))/len(line['evidence'])
                        else:
                            recall_score = float(sum([ev in line[eval_key + '_context'] for ev in line["evidence"]]))/len(line['evidence'])

                    answer_length = len(str(prediction).split())

                    cache_suffix = " [cache]" if from_cache else ""
                    print(f"Question {i+1}: Judge Score = {accuracy_score} ({accuracy_label}){cache_suffix}, Reasoning: {reasoning[:100]}...")
                    pbar.update(1)

                    return {
                        'score': accuracy_score,
                        'recall': recall_score,
                        'length': answer_length
                    }

                except Exception as e:
                    print(f"Error judging question {i+1}: {str(e)}. Defaulting to 0.0")
                    pbar.update(1)

                    return {
                        'score': 0.0,
                        'recall': 1.0,  # Default recall score
                        'length': 0     # Default length
                    }

        # Create tasks for all questions
        tasks = []
        for i, line in enumerate(qas):
            tasks.append(process_single_question(i, line))

        # Execute all tasks concurrently and collect results
        results = await asyncio.gather(*tasks)

        # Extract results from concurrent processing
        for result in results:
            all_scores.append(result['score'])
            all_recall.append(result['recall'])
            answer_lengths.append(result['length'])

    print(f"{len(qas)} QA samples evaluated by LLM judge; {len(all_scores)} accuracy values")
    lens = round(np.mean(answer_lengths), 4) if answer_lengths else 0.0
    return all_scores, lens, all_recall
