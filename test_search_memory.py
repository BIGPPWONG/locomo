from  langgraph_sdk import get_sync_client as get_client
from pprint import pprint

langgraph_config = {"user_id": "conv-26", "url": "http://localhost:8001", "graph_id": "agent", "max_concurrent_writes": 5, "ingest_model": ""}


url = langgraph_config.get('url', 'http://localhost:8001')
graph_id = langgraph_config.get('graph_id', 'agent')
user_id = langgraph_config.get('user_id', 'test_user')
max_concurrent = langgraph_config.get('max_concurrent_writes', 5)  # 默认最多5个并发写入

langgraph_client = get_client(
    url=url,
)
ingest_runnable_config = {"configurable":
    {
        "user_id": user_id,
        "run_mode": "ingest",
        "ingest_model": langgraph_config.get('ingest_model', '')
}
}
print("ingest_runnable_config:", ingest_runnable_config)



memory_namespace = ("chat", user_id, "memories")
m = langgraph_client.store.search_items(memory_namespace,
                                        # query="You're so strong and inspiring.",
                                        query="",
                                        filter={"index": {"$gte": "00", "$lt": "5"}},
)
pprint(m)