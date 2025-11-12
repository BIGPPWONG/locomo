"""
LangGraph/Assistant API integration for LoCoMo evaluation
"""
import json
import uuid
import random
import asyncio
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from langgraph_sdk import get_client as get_client_sdk
from langchain_core.messages import HumanMessage
import re
import tiktoken
from diskcache import Cache

config = {"configurable": {"user_id": "23fsddfgljdflg"},
          "metadata": {"langfuse_session_id": "essssssssssw"} }

# Global LangGraph database state for tracking processed conversations
langgraph_conversation_store = {}

# Cache for LangGraph answers keyed by question text
LANGGRAPH_CACHE_DIR = (Path(__file__).resolve().parent / ".." / "cache" / "langgraph_answers").resolve()
LANGGRAPH_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
LANGGRAPH_ANSWER_CACHE = Cache(str(LANGGRAPH_CACHE_DIR))


def log_with_time(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def should_cache_answer(answer):
    """
    Return True when the answer looks valid enough to store in cache.
    Filters out placeholders and error strings.
    """
    if not isinstance(answer, str):
        return False

    trimmed = answer.strip()
    if not trimmed:
        return False

    lowered = trimmed.lower()
    disallowed_tokens = (
        "error",
        "langgraph",
        "error occurred",
        "api error",
        "no response generated",
        "retrieve phase skipped"
    )

    return not any(token in lowered for token in disallowed_tokens)

# Token limit configurations (consistent with gpt_utils.py)
MAX_LENGTH = {
    'gpt-4-turbo': 128000,
    'gpt-4': 4096,
    'gpt-3.5-turbo': 16384,
    'gpt-3.5-turbo-16k': 16384,
    'gpt-4-32k': 32768,
    'gpt-4-turbo-2024-04-09': 128000,
    'gpt-4o': 128000,
    'gpt-4o-mini': 128000,
    'claude-3-5-sonnet': 200000,
    'claude-3-opus': 200000,
    'claude-3-haiku': 200000,
    'default': 32768,  # Default fallback
}
PER_QA_TOKEN_BUDGET = 50

# Prompts definition (consistent with gpt_utils.py)
CONV_START_PROMPT = (
    "You are an intelligent RAG agent analyzing a conversation between two people: {speaker1} and {speaker2}. "
    "The conversation spans multiple days, and information may appear in different contexts.\n\n"
    "Your goal is to accurately answer questions about this conversation by using the `search_memory` tool.\n\n"
    "### Search Memory Tool Usage\n"
    "- The `search_memory` tool performs semantic and contextual retrieval through the conversation history.\n"
    "- It returns a list of results, each formatted as a dictionary:\n"
    "  {\n"
    "    \"index\": <dialogue_index_number_as_zero_padded_string>,   # e.g., \"000\", \"001\", ..., \"253\"\n"
    "    \"date\": <conversation_date>,   # the date when this dialogue was spoken, NOT the event date\n"
    "    \"speaker\": <speaker_name>,\n"
    "    \"text\": <dialogue_text_content>,\n"
    "    \"blip_caption\": <image_description_if_present>  # empty string if no image\n"
    "  }\n\n"
    "- The `index` field is stored as a 3-digit zero-padded string (\"000\"–\"999\"), so lexicographic comparison equals numeric order.\n"
    "- Use the `query` parameter for semantic search by meaning.\n"
    "- Use the `filter` parameter to narrow results. Examples:\n"
    "  filter=\"{\\\"speaker\\\": \\\"Person1\\\"}\" → search only Person1's dialogue\n"
    "  filter=\"{\\\"index\\\": {\\\"$gt\\\": \\\"010\\\"}}\" → search dialogue after index '010'\n"
    "  filter=\"{\\\"speaker\\\": \\\"Person1\\\", \\\"index\\\": {\\\"$gte\\\": \\\"005\\\", \\\"$lte\\\": \\\"012\\\"}}\" → combine multiple conditions\n"
    "- Note: The `filter` parameter does not support regex operators like `$regex`.\n\n"
    "### Retrieval Strategy\n"
    "1. **Initial semantic search** — Start with `search_memory(query=\"<keywords>\")` to find semantically related dialogues.\n"
    "2. **Contextual expansion** — For each relevant result, use its `index` to gather neighboring context lines:\n"
    "   - Example: if result_index = \"013\", call `search_memory(query=\"\", filter={\"index\": {\"$gte\": \"010\", \"$lte\": \"016\"}})`.\n"
    "   - This retrieves 3 dialogues before and after the match for richer local context.\n"
    "3. **Aggregation** — Merge retrieved dialogues from all searches for full understanding.\n"
    "4. **Iteration and Exit Condition** —\n"
    "   - You may repeat searches to refine results, but perform **no more than 5 searches total**.\n"
    "   - Before every additional search, first evaluate: *“Is new information likely to change the final answer?”*\n"
    "   - If the retrieved context already contains sufficient clues, **stop searching and answer**.\n"
    "   - If no new relevant information is found after 2 consecutive searches, **stop and synthesize the best possible answer**.\n\n"
    "### Temporal Reasoning Rules\n"
    "- The `date` field represents when the dialogue **was spoken**, not when the described **event** occurred.\n"
    "- When asked about event time (e.g. \"When did it happen?\"), infer the event time logically from dialogue content and context — "
    "consider expressions like 'tomorrow', 'next week', 'yesterday', or explicit timestamps mentioned in the text.\n"
    "- Use relative reasoning: if Person1 said 'We’ll meet tomorrow' on 2024-08-01, infer that the meeting happens on 2024-08-02.\n"
    "- If multiple references appear across days, use later confirmations or contextual clues to determine the most likely actual event time.\n\n"
    "### Guidelines\n"
    "- Begin by checking whether you already have enough information to answer.\n"
    "- If any detail is missing or uncertain, call `search_memory` again (within the 5-search limit).\n"
    "- Prefer retrieving nearby dialogues by `index` rather than overly broad semantic queries.\n"
    "- When reasoning about time, never assume the `date` of a dialogue equals the event date unless explicitly stated.\n"
    "- Always aim to **provide a final answer once confidence is sufficient**, rather than continuing endless searches.\n"
    "- If still uncertain after the allowed searches, provide your best inferred answer and clearly mark uncertainty.\n\n"
    "### Output Format\n"
    "- Provide reasoning only if necessary, but always end your response with:\n"
    "  FINAL ANSWER: <short and direct answer only>\n"
    "- Keep FINAL ANSWER concise (preferably under 15 words) and avoid unnecessary explanation unless explicitly requested.\n"
)





def format_conv_start_prompt(speaker1, speaker2):
    """
    Safely inject speaker names into the conversation start prompt.
    """
    return CONV_START_PROMPT.replace("{speaker1}", speaker1).replace("{speaker2}", speaker2)





def preprocess_question_by_category(qa):
    """
    返回 (processed_question, allow_not_mentioned)
    """
    question = qa['question']
    category = qa.get('category', 1)

    if category == 2:
        return question + ' Use DATE of CONVERSATION to answer with an approximate date.', False
    elif category == 5:
        # 为 category 5 创建选择题格式，从 adversarial_answer 获取答案
        answer = qa.get('adversarial_answer', '') or qa.get('answer', 'Not mentioned in the conversation')
        question_template = question + " Select the correct answer: (a) {} (b) {}. "
        if random.random() < 0.5:
            return question_template.format('Not mentioned in the conversation', answer), True
        else:
            return question_template.format(answer, 'Not mentioned in the conversation'), True
    else:
        return question, False

def get_input_context(conversation_data, args=None):
    """
    从对话数据中提取上下文信息，不包含token计算
    返回列表格式，方便格式化处理
    数据结构：使用conversation_data['conversation']作为对话数据来源
    """
    context_items = []

    # 检查数捠结构，使用conversation key
    if 'conversation' not in conversation_data:
        log_with_time("Warning: No 'conversation' key found in conversation_data")
        raise ValueError("Invalid conversation_data format")

    conversation = conversation_data['conversation']

    # 获取所有会话编号，遵循gpt_utils.py的方式
    session_nums = []
    for k in conversation.keys():
        if 'session' in k and 'date_time' not in k:
            try:
                # 只处理标准格式的session key： session_1, session_2, 等
                parts = k.split('_')
                if len(parts) == 2 and parts[0] == 'session':
                    session_num = int(parts[1])
                    session_nums.append(session_num)
            except (ValueError, IndexError):
                # 跳过不符合标准格式的key
                continue

    if not session_nums:
        return context_items

    # 按会话顺序构建上下文
    for i in range(min(session_nums), max(session_nums) + 1):
        session_key = f'session_{i}'
        date_key = f'session_{i}_date_time'

        if session_key in conversation and date_key in conversation:
            session_context = {
                'date': conversation[date_key],
                'dialogues': []
            }

            # 处理当前会话的所有对话
            for dialog in conversation[session_key]:
                dialogue_item = {
                    'speaker': dialog['speaker'],
                    'text': dialog['text'],
                    'dia_id': dialog.get('dia_id', '')  # 可选的对话ID
                }
                if "blip_caption" in dialog:
                    dialogue_item['blip_caption'] = dialog["blip_caption"]
                session_context['dialogues'].append(dialogue_item)

            context_items.append(session_context)

    return context_items


async def prepare_rag_context(conversation_data, langgraph_config=None, args=None):
    """
    Prepare RAG context for LangGraph - 逐个message传入而不是将整个session格式化为一段文本

    Args:
        conversation_data: 对话数据
        langgraph_config: LangGraph配置信息，可以设置max_concurrent_writes来控制并发数（必须正确提供）
        args: 参数配置（可选）
    """
    log_with_time("Preparing RAG messages for LangGraph ingestion...")
    # 使用get_input_context获取内容列表
    context_items = get_input_context(conversation_data, args)

    # 获取说话人信息
    speakers = get_speaker_names(conversation_data)

    # 获取LangGraph client和配置
    sample_id = conversation_data.get('sample_id', 'unknown')
    assert langgraph_config is not None, "langgraph_config must be provided to get LangGraph client"

    bearer = langgraph_config.get('bearer', 'Bearer gss_token')
    host = langgraph_config.get('host', 'localhost')
    url = langgraph_config.get('url', 'http://localhost:8001')
    graph_id = langgraph_config.get('graph_id', 'agent')
    max_concurrent = langgraph_config.get('max_concurrent_writes', 5)  # 默认最多5个并发写入

    langgraph_client = get_client_sdk(
        url=url,
        headers={
            "Authorization": bearer,
            "Host": host
        }
    )
    ingest_runnable_config = {"configurable":
        {
            "user_id": sample_id,
            "run_mode": "ingest",
            "ingest_model": langgraph_config.get('ingest_model', '')
    }
    }
    log_with_time(f"ingest_runnable_config: {ingest_runnable_config}")

    sem = asyncio.Semaphore(max_concurrent)

    # 计算总消息数用于进度显示
    total_messages = sum(len(session['dialogues']) for session in context_items)

    # 使用tqdm显示进度
    with tqdm(total=total_messages, desc=f"Ingesting RAG messages to LangGraph (max {max_concurrent} concurrent)") as pbar:
        async def process_message(session_idx, session_date, dialogue):
            async with sem:
                try:
                    # 构建单个消息的RAG内容
                    speaker = dialogue['speaker']
                    text = dialogue['text']
                    blip_caption = dialogue.get('blip_caption', '')
                    dia_id = dialogue.get('dia_id', '')

                    # 创建结构化的消息内容，包含日期和对话信息
                    message_content = f"DATE: {session_date}\n{dia_id}: {speaker} said: \"{text}\""
                    if blip_caption:
                        message_content += f" and shared {blip_caption}"

                    memory_namespace = ("chat", sample_id, "memories")

                    # 处理 dia_id，提取数字序号并归一化为 3 位数字字符串
                    if dia_id and ":" in dia_id:
                        try:
                            # 提取冒号后的数字部分，如 "D1:1" -> "1"
                            seq_num = int(dia_id.split(":")[-1])
                            # 归一化为 3 位数字字符串，如 "1" -> "001"
                            normalized_index = f"{seq_num:03d}"
                        except (ValueError, IndexError):
                            # 如果转换失败，使用原始 dia_id
                            normalized_index = str(dia_id)
                    elif dia_id:
                        # 如果没有冒号但 dia_id 存在，尝试转换为数字并归一化
                        try:
                            seq_num = int(dia_id)
                            normalized_index = f"{seq_num:03d}"
                        except ValueError:
                            # 如果无法转换为数字，使用原始值
                            normalized_index = str(dia_id)
                    else:
                        # 如果 dia_id 为空，生成一个唯一的索引
                        import random
                        normalized_index = f"{random.randint(0, 999):03d}"

                    m = await langgraph_client.store.put_item(memory_namespace,
                                                     key=str(uuid.uuid4()),
                                                     value={
                                                         "index": normalized_index,  # 存储归一化的字符串索引
                                                         "date": session_date,
                                                         "speaker": speaker,
                                                         "text": text,
                                                         "blip_caption": blip_caption,
                                                         "original_dia_id": dia_id,  # 保留原始 dia_id 用于调试
                                                     },
                                                     index=["speaker", "text", "blip_caption"],)

                    # run = await langgraph_client.runs.wait(
                    #     None, graph_id,
                    #     input={"messages": [{"role": "user", "content": message_content}]},
                    #     config=ingest_runnable_config,
                    #     on_completion="delete"
                    # )
                    pbar.update(1)

                except Exception as e:
                    pbar.set_postfix({"error": f"session {session_idx+1}"})
                    log_with_time(f"Error storing RAG message for {sample_id} session {session_idx+1}: {str(e)}")

        # 批量提交所有消息
        tasks = []
        for i, session in enumerate(context_items):
            session_date = session['date']
            for dialogue in session['dialogues']:
                tasks.append(process_message(i, session_date, dialogue))

        await asyncio.gather(*tasks)

    log_with_time(f"Completed RAG context storage for {sample_id}")

    return

def format_context_to_text(context_items, reverse_dialogues=False, args=None):
    """
    将列表格式的上下文转换为文本格式，始终执行token控制

    Args:
        context_items: get_input_context返回的列表
        reverse_dialogues: 是否反转对话顺序（优先显示最新内容）
        args: 包含模型参数的配置对象，用于token控制（可选）

    Returns:
        格式化后的文本上下文，确保不会超出模型的token限制
    """
    # 初始化token编码器
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            encoding = None

    # 获取token限制配置
    if args is not None:
        model = getattr(args, 'model', 'gpt-4')
    else:
        model = 'gpt-4'  # 默认模型

    max_tokens = MAX_LENGTH.get(model, MAX_LENGTH['default'])

    # 预留8000 tokens作为prompt和answer的空间
    reserved_tokens = 8000
    target_max_tokens = max_tokens - reserved_tokens

    # 确保target_max_tokens至少为正数
    target_max_tokens = max(target_max_tokens, 1000)

    # 逆向构建文本内容（优先保留最新内容）
    if reverse_dialogues:
        # 如果要反转对话顺序，我们逆向遍历session和dialogue
        ordered_sessions = context_items[::-1]
    else:
        # 正常顺序，但为了token控制，我们逆向处理但最后按顺序输出
        ordered_sessions = context_items

    text_segments = []
    current_tokens = 0

    # 逆向处理sessions
    for session in reversed(ordered_sessions):
        session_text_parts = []

        # session头部
        session_header = f"DATE: {session['date']}\nCONVERSATION:\n"
        session_text_parts.append(session_header)

        # 获取对话列表
        dialogues = session['dialogues']
        if reverse_dialogues:
            dialogues_to_process = dialogues[::-1]
        else:
            # 为了token控制，我们从最新对话开始处理
            dialogues_to_process = dialogues[::-1]

        session_dialogue_texts = []

        for dialogue in dialogues_to_process:
            line = f"{dialogue['speaker']} said, \"{dialogue['text']}\""
            if 'blip_caption' in dialogue:
                line += f" and shared {dialogue['blip_caption']}"
            session_dialogue_texts.append(line)

        # 如果是正常顺序，需要反转对话文本以保持时间顺序
        if not reverse_dialogues:
            session_dialogue_texts = session_dialogue_texts[::-1]

        session_text_parts.extend(session_dialogue_texts)
        session_text_parts.append("")  # 会话间隔

        # 计算整个session的token数量
        session_text = '\n'.join(session_text_parts)

        if encoding:
            try:
                session_tokens = len(encoding.encode(session_text))
            except:
                # 编码失败时估算
                session_tokens = len(session_text) // 4
        else:
            session_tokens = len(session_text) // 4

        # 检查是否超出限制
        if current_tokens + session_tokens <= target_max_tokens:
            # 可以添加整个session
            current_tokens += session_tokens
            # 插入到开头以保持正确的顺序
            text_segments.insert(0, session_text)
        else:
            # 部分添加或跳过
            if current_tokens >= target_max_tokens:
                break  # 已达限制，停止添加

            # 尝试逐个添加对话直到达到限制
            remaining_tokens = target_max_tokens - current_tokens
            header_text = f"DATE: {session['date']}\nCONVERSATION:\n"

            if encoding:
                try:
                    header_tokens = len(encoding.encode(header_text))
                except:
                    header_tokens = len(header_text) // 4
            else:
                header_tokens = len(header_text) // 4

            if header_tokens >= remaining_tokens:
                break  # 连头部都加不进去，停止

            partial_session_parts = [header_text]
            used_tokens = header_tokens
            remaining_tokens -= header_tokens

            # 逐个添加对话
            for dialogue in session_dialogue_texts:
                if encoding:
                    try:
                        dialogue_tokens = len(encoding.encode(dialogue))
                    except:
                        dialogue_tokens = len(dialogue) // 4
                else:
                    dialogue_tokens = len(dialogue) // 4

                if used_tokens + dialogue_tokens <= target_max_tokens:
                    partial_session_parts.append(dialogue)
                    used_tokens += dialogue_tokens
                else:
                    break

            if len(partial_session_parts) > 1:  # 至少有头部
                text_segments.insert(0, '\n'.join(partial_session_parts))
            break  # 达到限制，停止

    # 如果没有添加任何内容，返回空字符串
    if not text_segments:
        return ""

    return '\n'.join(text_segments)



def get_speaker_names(conversation_data):
    """
    从对话数据中提取说话人姓名
    """
    speakers = set()
    if 'conversation' in conversation_data and 'session_1' in conversation_data['conversation']:
        for entry in conversation_data['conversation']['session_1']:
            if 'speaker' in entry:
                speakers.add(entry['speaker'])
    return list(speakers) if speakers else ["Person1", "Person2"]

def get_langgraph_answers(data, out_data, prediction_key, args, langgraph_config):
    """
    同步包装函数，用于保持API兼容性
    内部调用异步的并发版本来处理答案检索
    """
    import asyncio
    return asyncio.run(get_langgraph_answers_async(data, out_data, prediction_key, args, langgraph_config))

async def get_langgraph_answers_async(data, out_data, prediction_key, args, langgraph_config):
    """
    通过 LangGraph SDK 调用 Assistant API 获取答案，支持 start_prompt 和 category-based 预处理
    """
    # 检查是否已有预测结果
    if prediction_key in out_data['qa'][0]:
        return out_data

    # 根据参数决定是否执行RAG上下文注入
    if not args.skip_langgraph_rag:
        # 使用prepare_rag_context准备RAG上下文，传入langgraph_config以获取client
        rag_context = await prepare_rag_context(data, langgraph_config, args)
    else:
        log_with_time(f"Skipping RAG context injection for sample {data.get('sample_id', 'unknown')}")
        rag_context = None

    # Check if retrieve phase should be skipped
    if args.skip_langgraph_retrieve:
        log_with_time(f"Skipping LangGraph retrieve phase for sample {data.get('sample_id', 'unknown')}")
        # Generate placeholder answers for all questions
        predictions = ["Retrieve phase skipped - no answer generated"] * len(data['qa'])

        # Store placeholder predictions in out_data
        for i, qa in enumerate(out_data['qa']):
            qa[prediction_key] = predictions[i]

        return out_data

    # 获取对话者名称并创建 start_prompt
    speakers = get_speaker_names(data)
    speaker_defaults = ["Person1", "Person2"]
    speaker1, speaker2 = (speakers + speaker_defaults)[:2]
    start_prompt = format_conv_start_prompt(speaker1, speaker2)

    # 获取并格式化对话历史上下文
    conversation_context = get_input_context(data, args)
    formatted_context = format_context_to_text(conversation_context, reverse_dialogues=False, args=args)

    # # 获取上下文信息
    # context = data.get('context', '')
    # if context:
    #     start_prompt += f"Context: {context}\n\n"

    # 从config中获取连接参数
    bearer = langgraph_config.get('bearer', 'Bearer gss_key')
    host = langgraph_config.get('host', 'localhost')
    url = langgraph_config.get('url', 'http://localhost:8001')
    graph_id = langgraph_config.get('graph_id', 'agent')
    sample_id = data.get('sample_id', 'unknown')
    langgraph_config.update({
        'user_id': sample_id,
    })

    # 获取对话配置
    user_id = data.get('sample_id', 'unknown')

    predictions = []

    # Extract max_concurrent setting from langgraph_config, default to 5
    max_concurrent = langgraph_config.get('max_concurrent_writes', 5)

    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(max_concurrent)

    # Process all questions concurrently with progress tracking
    total_questions = len(data['qa'])
    with tqdm(total=total_questions, desc=f"Getting LangGraph answers (max {max_concurrent} concurrent)") as pbar:
        # Define async function to process a single question
        async def process_question(i, qa):
            async with sem:
                try:
                    # 根据 category 预处理问题
                    original_question = qa['question']
                    category = str(qa.get('category', 'unknown'))
                    cache_key = f"{sample_id}::{category}::{original_question}"

                    # 返回缓存命中以避免重复调用
                    cached_answer = LANGGRAPH_ANSWER_CACHE.get(cache_key)
                    if cached_answer is not None:
                        log_with_time(f"Cache hit for question {i+1}: {original_question} -> Answer: {cached_answer}")
                        pbar.update(1)
                        return cached_answer

                    processed_question, allow_not_mentioned = preprocess_question_by_category(qa)

                    # 当 category 为 5 时不注入上下文
                    category_value = qa.get('category', 1)
                    if category_value == 5:
                        # Category 5 不包含上下文
                        full_system_prompt = start_prompt + (
                            "\n\nAnswer rules:\n"
                            "- Keep answer concise and factual. End with 'FINAL ANSWER: <answer>'.\n"
                            "- Do not include reasoning in FINAL ANSWER. If you must include reasoning, place it before FINAL ANSWER.\n"
                        )
                    else:
                        # 其他 categories 包含对话历史上下文
                        full_system_prompt = start_prompt + (
                            "\n\nConversation Context:\n"
                            f"{formatted_context}\n\n"
                            "Answer rules:\n"
                            "- Keep answer concise and factual. End with 'FINAL ANSWER: <answer>'.\n"
                            "- Do not include reasoning in FINAL ANSWER. If you must include reasoning, place it before FINAL ANSWER.\n"
                        )

                    # 根据该题是否允许 Not mentioned，动态加入约束
                    if allow_not_mentioned:
                        full_system_prompt += "\nThis question may include 'Not mentioned in the conversation' as a valid choice. You may output that option if appropriate."
                    else:
                        full_system_prompt += "\nDo NOT output 'Not mentioned in the conversation' unless you have exhaustively searched the conversation store and confirmed there is zero evidence. For this dataset, prefer giving a direct answer based on retrieved context."


                    # 执行 LangGraph 调用
                    answer = await async_langgraph_call(
                        system_prompt=full_system_prompt,
                        user_message=processed_question,
                        config=langgraph_config
                    )

                    # 提取和处理答案
                    processed_answer = extract_short_answer(answer, processed_question)
                    log_with_time(f"Processed question {i+1}: {original_question} -> Answer: {processed_answer}")
                    if should_cache_answer(answer):
                        LANGGRAPH_ANSWER_CACHE[cache_key] = processed_answer
                    pbar.update(1)
                    return processed_answer

                except Exception as e:
                    log_with_time(f"Error processing question {i+1}: {str(e)}")
                    pbar.set_postfix({"error": f"question {i+1}"})
                    pbar.update(1)
                    return "Error occurred"

        # Create tasks for all questions
        tasks = []
        for i, qa in enumerate(data['qa']):
            tasks.append(process_question(i, qa))

        # Execute all tasks concurrently and collect results
        results = await asyncio.gather(*tasks)
        predictions = list(results)

    # 存储预测结果
    log_with_time(f"Generated {len(predictions)} predictions")
    for i, qa in enumerate(out_data['qa']):
        if i < len(predictions):
            qa[prediction_key] = predictions[i]
        else:
            qa[prediction_key] = ""

    return out_data


def extract_short_answer(ai_content, question):
    """
    从AI回复中提取格式化的最终答案 (支持 'FINAL ANSWER:' 格式)
    """
    content = ai_content.strip()

    # 1️⃣ 优先提取 FINAL ANSWER 标记段
    match = re.search(r'FINAL ANSWER[:：]\s*(.*)', content, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # 去掉多余标点或换行
        answer = re.split(r'[\n\r]', answer)[0].strip()
        # 裁剪过长内容
        return answer[:400]

    # 2️⃣ 如果没有标记，回退到传统句子提取逻辑
    sentences = re.split(r'[.!。]', content)
    for s in sentences:
        s = s.strip()
        if 5 < len(s) < 80:
            return s

    return content[:400]


# API配置函数
def call_langgraph_cloud_api(system_prompt, user_message, config):
    """LangGraph Cloud API 实现 - 异步版本"""
    # 这里使用上面实现的完整流程
    return asyncio.run(async_langgraph_call(system_prompt, user_message, config))


async def async_langgraph_call(system_prompt, user_message, config):
    """异步执行 LangGraph 调用"""
    try:
        # 获取配置
        bearer = config.get('bearer', 'Bearer gss_token')
        host = config.get('host', 'localhost')
        url = config.get('url', 'http://localhost:8001')
        graph_id: str = config.get('graph_id', 'agent')

        client = get_client_sdk(
            url=url,
            headers={
                "Authorization": bearer,
                "Host": host
            }
        )

        # 会话配置
        user_id = config.get('user_id', 'unknown')
        session_id = config.get('session_id', str(uuid.uuid4()))

        retrieve_runnable_config = {"configurable":
            {
                "user_id": user_id,
                "run_mode": "retrieve",  # or "ingest"
                "retrieve_model": config.get('retrieve_model', '')
            },
            "recursion_limit": 60
        }

        # 准备输入
        input_data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }

        # 创建线程和执行
        thread_id = str(uuid.uuid4())
        thread = await client.threads.create(thread_id=thread_id)

        # assistant = await client.assistants.create(
        #     graph_id=graph_id,
        #     name="LoCoMo_QA_Assistant"
        # )
        ai_message = None
        # 使用 wait 模式获取完整响应
        # run = await client.runs.wait(
        #     thread_id,
        #     graph_id,
        #     input=input_data,
        #     config=retrieve_runnable_config,
        #     on_completion="delete"  # stateless模式完成后删除资源
        # )
        # ai_message = run.get("messages", [])[-1].get("content", "")

        # 使用 stream 模式获取完整响应
        async for chunk in client.runs.stream(
            thread_id,
            graph_id,
            input=input_data,
            config=retrieve_runnable_config,
            stream_mode=["messages"],
            on_completion="delete"  # stateless模式完成后删除资源
        ):
            run = chunk  # 最终的完整响应存储在 run 变量中
        ai_message = run.data[-1].get("content", "")
        
        
        log_with_time(f"AI Message: {ai_message}")

        return ai_message if ai_message else "No response generated"

    except Exception as e:
        log_with_time(f"LangGraph API call error: {str(e)}")
        return f"API Error: {str(e)}"




# 配置实用函数
def create_langgraph_config(api_type='mock', **kwargs):
    """Create a LangGraph configuration object"""
    config = {'api_type': api_type}
    config.update(kwargs)
    return config


# 后向兼容函数调用（保持原有接口）
def call_langgraph_api(system_prompt, user_message, config):
    """
    Make the actual LangGraph/Assistant API call

    This is where you'll implement the actual API integration.
    Options include:
    - LangGraph Cloud API
    """

    # LangGraph Cloud API integration
    return call_langgraph_cloud_api(system_prompt, user_message, config)
