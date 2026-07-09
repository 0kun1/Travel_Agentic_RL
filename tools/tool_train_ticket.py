# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 火车票查询工具
# --------------------------------------------

import asyncio
import os
import json
import logging
from pathlib import Path

from openai import AsyncOpenAI
from dotenv import load_dotenv

from prompt import TRAIN_TICKET_SYSTEM_PROMPT
from qwen_agent.tools.base import BaseTool, register_tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


@register_tool("train_tickets_search", allow_overwrite=True)
class TrainTicketsSearch(BaseTool):
    name = "train_tickets_search"
    description = "根据日期查询城市间火车/动车/高铁票信息。"

    parameters = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "日期，格式 YYYY-MM-DD"
            },
            "from_city": {
                "type": "string",
                "description": "出发城市中文名"
            },
            "to_city": {
                "type": "string",
                "description": "到达城市中文名"
            },
        },
        "required": ["date", "from_city", "to_city"]
    }

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)

    async def call(self, params: dict, **kwargs) -> str:
        try:
            date = params["date"]
            from_city = params["from_city"]
            to_city = params["to_city"]
        except Exception:
            return "[train_tickets_search] Invalid request format: Input must be a JSON object containing 'date', 'from_city', and 'to_city'."

        query = json.dumps(
            {
                "date": date,
                "from_city": from_city,
                "to_city": to_city,
            },
            ensure_ascii=False,
        )

        messages = [
            {"role": "system", "content": TRAIN_TICKET_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        result = None

        async with AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
            timeout=60,
            max_retries=2,
        ) as client:
            try:
                response = await client.chat.completions.create(
                    model=os.environ["LLM_MODEL_ID"],
                    messages=messages,
                )
                result = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"[train_tickets_search] Failed to get response: {e}")
                result = f"[train_tickets_search] Failed to get response: {params}"

        if not result:
            result = "两地无直达火车票"

        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    search = TrainTicketsSearch()
    params = {
        "date": "2026-03-30",
        "from_city": "成都",
        "to_city": "北京",
    }

    res = asyncio.run(search.call(params))
    print(res)