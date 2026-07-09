# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 网页访问工具
# --------------------------------------------

import sys
import os
import json
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from firecrawl import Firecrawl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from prompt import EXTRACTOR_PROMPT
from utils.text import truncate_text
from qwen_agent.tools.base import BaseTool, register_tool


logger = logging.getLogger(__name__)

VISIT_MAX_CONTENT_LEN = int(os.getenv("VISIT_MAX_CONTENT_LEN", "12000"))
VISIT_SUMMARY_MODEL = os.getenv("LLM_MODEL_ID", "deepseek-chat")


def _to_dict(obj):
    """
    Firecrawl 返回对象可能是 dict，也可能是 pydantic model。
    统一转成 dict。
    """
    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        return {
            k: v
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }

    return {}


@register_tool("visit", allow_overwrite=True)
class Visit(BaseTool):
    name = "visit"
    description = "访问网页并根据目标信息返回内容摘要。"

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要访问的网页 URL。"
            },
            "goal": {
                "type": "string",
                "description": "访问网页需要获得的目标信息。"
            },
        },
        "required": ["url", "goal"],
    }

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)

        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        if not firecrawl_api_key:
            raise RuntimeError("Missing FIRECRAWL_API_KEY in .env")

        self.firecrawl_client = Firecrawl(api_key=firecrawl_api_key)

    def _scrape_once(self, url: str) -> str:
        """
        使用 Firecrawl 抓取网页内容。
        优先取 markdown，其次取 html / text。
        """
        result = self.firecrawl_client.scrape(url)

        data = _to_dict(result)

        markdown = data.get("markdown")
        html = data.get("html")
        text = data.get("text")

        content = markdown or text or html or ""

        if not content:
            return ""

        return truncate_text(content, max_len=VISIT_MAX_CONTENT_LEN)

    async def scrape(self, url: str) -> str:
        try:
            return await asyncio.to_thread(self._scrape_once, url)
        except Exception as e:
            logger.error(f"[Visit] scrape failed: {e}")
            return ""

    async def extract_with_llm(self, webpage_content: str, goal: str) -> str:
        prompt = EXTRACTOR_PROMPT.format(
            webpage_content=webpage_content,
            goal=goal,
        )

        messages = [
            {
                "role": "system",
                "content": "你是一个严谨的网页信息抽取助手。请根据网页内容和目标提取有用信息。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        async with AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
            timeout=60,
            max_retries=2,
        ) as client:
            response = await client.chat.completions.create(
                model=VISIT_SUMMARY_MODEL,
                messages=messages,
            )

        return response.choices[0].message.content.strip()

    async def call(self, params: dict, **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except Exception:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        webpage_content = await self.scrape(url)

        if not webpage_content:
            return f"[Visit] Failed to scrape webpage content from url: {url}"

        try:
            extracted = await self.extract_with_llm(webpage_content, goal)
        except Exception as e:
            logger.error(f"[Visit] extract failed: {e}")
            return (
                f"[Visit] Failed to extract information from webpage. "
                f"Raw webpage content preview:\n{truncate_text(webpage_content, max_len=3000)}"
            )

        result = {
            "url": url,
            "goal": goal,
            "extracted": extracted,
        }

        return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tool = Visit()

    params = {
        "url": "https://cd.bendibao.com/tour/2019918/104907.shtm",
        "goal": "提取成都三日游路线、景点安排和交通建议。",
    }

    res = asyncio.run(tool.call(params))
    print(res)