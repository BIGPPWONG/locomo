"""
LangGraph/Assistant API integration for LoCoMo evaluation
"""
import json
import uuid
import random
import asyncio
from tqdm import tqdm
from langgraph_sdk import get_client as get_client_sdk
from langchain_core.messages import HumanMessage


config = {"configurable": {"user_id": "23fsddfgljdflg"},
          "metadata": {"langfuse_session_id": "essssssssssw"} }

# Global LangGraph database state for tracking processed conversations
langgraph_conversation_store = {}

# Prompts definition (consistent with gpt_utils.py)
CONV_START_PROMPT = (
    "You will be asked questions about a conversation between two people: {} and {}. "
    "The conversation takes place over multiple days.\n\n"
    "When answering, you must use the `search_memory` tool to look up information related "
    "to this conversation before providing your response."
)
def preprocess_question_by_category(qa):
    """
    根据问题category预处理问题文本，与gpt_utils.py保持一致
    传入完整的qa字典，从qa['adversarial_answer']获取category 5的答案
    """
    question = qa['question']
    category = qa.get('category', 1)

    if category == 2:
        return question + ' Use DATE of CONVERSATION to answer with an approximate date.'
    elif category == 5:
        # 为category 5创建选择题格式，从adversarial_answer获取答案
        answer = qa.get('adversarial_answer', '')
        if not answer:
            # 如果没有adversarial_answer，使用标准answer
            answer = qa.get('answer', 'Not mentioned in the conversation')

        question_template = question + " Select the correct answer: (a) {} (b) {}. "
        if random.random() < 0.5:
            return question_template.format('Not mentioned in the conversation', answer)
        else:
            return question_template.format(answer, 'Not mentioned in the conversation')
    else:
        return question

def get_input_context(conversation_data, args=None):
    """
    从对话数据中提取上下文信息，不包含token计算
    返回列表格式，方便格式化处理
    数据结构：使用conversation_data['conversation']作为对话数据来源
    """
    context_items = []

    # 检查数捠结构，使用conversation key
    if 'conversation' not in conversation_data:
        print("Warning: No 'conversation' key found in conversation_data")
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
                    'text': dialog['text']
                }
                if "blip_caption" in dialog:
                    dialogue_item['blip_caption'] = dialog["blip_caption"]
                session_context['dialogues'].append(dialogue_item)

            context_items.append(session_context)

    return context_items


async def prepare_rag_context(conversation_data, langgraph_config=None, args=None):
    """
    Prepare RAG context for LangGraph - 使用tqdm步长控制并发数量

    Args:
        conversation_data: 对话数据
        langgraph_config: LangGraph配置信息，可以设置max_concurrent_writes来控制并发数（必须正确提供）
        args: 参数配置（可选）
    """
    print("Preparing RAG context for LangGraph ingestion...")
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
    print("ingest_runnable_config:", ingest_runnable_config)

    sem = asyncio.Semaphore(max_concurrent)
    # 使用tqdm显示进度
    with tqdm(total=len(context_items), desc=f"Ingesting RAG context to LangGraph (max {max_concurrent} concurrent)") as pbar:
        async def process_session(i, session):
            async with sem:
                rag_context = format_context_to_text([session], reverse_dialogues=True)
                try:
                    run = await langgraph_client.runs.wait(
                        None, graph_id,
                        input={"messages": [{"role": "user", "content": rag_context}]},
                        config=ingest_runnable_config,
                        on_completion="delete"
                    )
                    pbar.update(1)
                    print(f"Stored RAG context for {sample_id} session {i+1}")
                except Exception as e:
                    pbar.set_postfix({"error": f"session {i+1}"})
                    print(f"Error storing RAG context for {sample_id} session {i+1}: {str(e)}")
        # 批量提交
        tasks = [process_session(i, s) for i, s in enumerate(context_items)]
        await asyncio.gather(*tasks)

    print(f"Completed RAG context storage for {sample_id}")

    return

def format_context_to_text(context_items, reverse_dialogues=False):
    """
    将列表格式的上下文转换为文本格式

    Args:
        context_items: get_input_context返回的列表
        reverse_dialogues: 是否反转对话顺序（优先显示最新内容）

    Returns:
        格式化后的文本上下文
    """
    text_parts = []

    for session in context_items:
        text_parts.append(f"DATE: {session['date']}")
        text_parts.append("CONVERSATION:")

        dialogues = session['dialogues']
        if reverse_dialogues:
            dialogues = list(reversed(dialogues))

        for dialogue in dialogues:
            line = f"{dialogue['speaker']} said, \"{dialogue['text']}\""
            if 'blip_caption' in dialogue:
                line += f" and shared {dialogue['blip_caption']}"
            text_parts.append(line)

        text_parts.append("")  # 会话间隔

    return '\n'.join(text_parts)

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
    通过 LangGraph SDK 调用 Assistant API 获取答案，支持 start_prompt 和 category-based 预处理
    """
    # 检查是否已有预测结果
    if prediction_key in out_data['qa'][0]:
        return out_data

    # 根据参数决定是否执行RAG上下文注入
    if not args.skip_langgraph_rag:
        # 使用prepare_rag_context准备RAG上下文，传入langgraph_config以获取client
        rag_context = asyncio.run(prepare_rag_context(data, langgraph_config, args))
    else:
        print(f"Skipping RAG context injection for sample {data.get('sample_id', 'unknown')}")
        rag_context = None

    # 获取对话者名称并创建 start_prompt
    speakers = get_speaker_names(data)
    start_prompt = CONV_START_PROMPT.format(speakers[0], speakers[1]) if len(speakers) >= 2 else CONV_START_PROMPT.format("Person1", "Person2")

    # # 获取上下文信息
    # context = data.get('context', '')
    # if context:
    #     start_prompt += f"Context: {context}\n\n"

    # 从config中获取连接参数
    bearer = langgraph_config.get('bearer', 'Bearer gss_key')
    host = langgraph_config.get('host', 'localhost')
    url = langgraph_config.get('url', 'http://localhost:8001')
    graph_id = langgraph_config.get('graph_id', 'agent')

    # 获取对话配置
    user_id = data.get('sample_id', 'unknown')

    predictions = []

    # 只处理第一个问题用于测试
    for i, qa in enumerate(tqdm(data['qa'][:], desc="Getting LangGraph answers")):
        try:
            # 根据 category 预处理问题
            original_question = qa['question']
            processed_question = preprocess_question_by_category(qa)

            # 构建完整的系统提示，包含 start_prompt
            full_system_prompt = start_prompt + "Answer questions based on the provided conversation with exact words when possible."

            # 执行 LangGraph 调用
            answer = asyncio.run(async_langgraph_call(
                system_prompt=full_system_prompt,
                user_message=processed_question,
                config=langgraph_config
            ))

            # 提取和处理答案
            processed_answer = extract_short_answer(answer, processed_question)
            predictions.append(processed_answer)
            print(f"Processed question {i+1}: {original_question} -> Answer: {processed_answer}")

        except Exception as e:
            print(f"Error processing question {i+1}: {str(e)}")
            predictions.append("Error occurred")

    # 存储预测结果
    print(f"Generated {len(predictions)} predictions")
    for i, qa in enumerate(out_data['qa']):
        if i < len(predictions):
            qa[prediction_key] = predictions[i]
        else:
            qa[prediction_key] = ""

    return out_data


def extract_short_answer(ai_content, question):
    """
    从AI回复中提取简短答案
    """
    content = ai_content.strip()

    # 如果内容已经比较简短，直接返回
    if len(content) < 50:
        return content

    # 尝试提取最相关的部分
    # 移除前导的系统提示部分
    if "LLM 分析" in content:
        # 分割典型回答格式
        parts = content.split("\n")
        relevant_parts = []
        for part in parts:
            if not (part.startswith("用户说") or part.startswith("LLM 分析")):
                relevant_parts.append(part)

        if relevant_parts and "天气信息" in relevant_parts[-1]:
            return relevant_parts[-1].split(":")[-1].strip()

        if relevant_parts:
            return relevant_parts[-1]

    # 如果找不到合适的格式，返回主要内容的部分
    sentences = content.split(".")
    for sentence in sentences:
        if len(sentence.strip()) > 5:
            return sentence.strip()

    return content[:100] + "..." if len(content) > 100 else content


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
            }
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
        # thread_id = str(uuid.uuid4())
        # thread = await client.threads.create(thread_id=thread_id)

        # assistant = await client.assistants.create(
        #     graph_id=graph_id,
        #     name="LoCoMo_QA_Assistant"
        # )

        # 使用 wait 模式获取完整响应
        run = await client.runs.wait(
            None,
            graph_id,
            input=input_data,
            config=retrieve_runnable_config,
            on_completion="delete"  # stateless模式完成后删除资源
        )
        # 提取AI回复
        ai_message = None
        ai_message = run.get("messages", [])[-1].get("content", "")
        print(f"AI Message: {ai_message}")

        return ai_message if ai_message else "No response generated"

    except Exception as e:
        print(f"LangGraph API call error: {str(e)}")
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