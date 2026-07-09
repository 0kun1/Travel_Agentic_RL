import os
import re
import random
import time
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# 1. 加载环境变量
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

random.seed(42)


# =========================
# 2. 样本泛化 Prompt
# =========================

QUERY_EXPANSION_PROMPT = """
你是一个经验丰富的文字改写专家，对中国的旅游城市，交通，景点路线等非常熟悉。你的任务是根据输入的用户关于旅游的问题，扩充生成5个新的问题。

以下是一些生成新问题的要求：
1.给的用户问题只是参考，要输出完全新的问题，不能跟原有的句子意思相同或相近。
2.新的问题里面出现的地名，景点，必须是真实的，问题是符合逻辑的，请结合你的只旅游知识来进行改写，可以适当替换里面出现的地点/城市。
3.(可选)如果有必要，可以在提问里面增加一些约束，例如价格，距离，评分，交通方式，预算，途经点等。

请直接输出改写后的句子，每一个句子是一行，不要输出理由或者其他无关内容。

用户问题：{}
"""


# =========================
# 3. 创建 OpenAI-compatible client
# =========================

llm_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)


# =========================
# 4. 单条样本扩展函数
# =========================

def chat(prompt, max_retry=3):
    def do_chat(text):
        user_prompt = QUERY_EXPANSION_PROMPT.format(text)

        completions = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL_ID", "deepseek-v4-pro"),
            messages=[
                {"role": "system", "content": "你是有用的人工智能助手。"},
                {"role": "user", "content": user_prompt},
            ],
        )

        return text, completions.choices[0].message.content

    while max_retry > 0:
        try:
            return do_chat(prompt)
        except Exception as e:
            print(f"[WARN] chat failed, retry left={max_retry - 1}, error={e}")
            max_retry -= 1
            sleep_seconds = random.randint(1, 4)
            time.sleep(sleep_seconds)

    return None


# =========================
# 5. 主流程：读取种子数据，批量扩展
# =========================

if __name__ == "__main__":
    MAX_WORKERS = 50

    input_path = "./data/train_seed.jsonl"
    output_path = "./data/train_expansion.jsonl"

    queries = []
    with open(input_path, "r", encoding="utf-8") as fd:
        for idx, line in enumerate(fd):
            info = json.loads(line)
            queries.append((idx, info["question"]))

    expand_queries = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            idx: executor.submit(chat, query)
            for idx, query in queries
        }

        for idx in tqdm(futures):
            future = futures[idx]
            result = future.result()

            if result is None:
                continue

            query, expand = result

            expand = expand.split("\n")
            expand = [
                re.sub(r"^\s*\d+[\.\、\)]\s*", "", item).strip()
                for item in expand
                if item.strip()
            ]

            expand_queries.append(query)
            expand_queries.extend(expand)

    # 去重
    expand_queries = list(set(exp_queries)) if False else list(set(expand_queries))

    with open(output_path, "w", encoding="utf-8") as fw:
        for query in expand_queries:
            info = {"question": query}
            fw.write(json.dumps(info, ensure_ascii=False) + "\n")

    print(f"input queries: {len(queries)}")
    print(f"expanded queries: {len(expand_queries)}")
    print(f"saved to: {output_path}")