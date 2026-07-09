# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 高德天气查询工具
# --------------------------------------------

import sys
import os
import json
import logging
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from qwen_agent.tools.base import BaseTool, register_tool
from utils.markdown import json2md


logger = logging.getLogger(__name__)


@register_tool("weather_search", allow_overwrite=True)
class WeatherSearch(BaseTool):
    name = "weather_search"
    description = "根据城市名称查询指定城市天气。"

    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，例如 北京、上海、成都、广州。",
            }
        },
        "required": ["city"],
    }

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.api_key = os.getenv("AMAP_MAPS_API_KEY")

    async def call(self, params: dict, **kwargs) -> str:
        city = params.get("city")

        if not city:
            return "[weather_search] Invalid request format: missing city."

        if not self.api_key:
            return "[weather_search] Missing AMAP_MAPS_API_KEY in .env."

        url = "https://restapi.amap.com/v3/weather/weatherInfo"

        request_params = {
            "key": self.api_key,
            "city": city,
            "extensions": "all",
            "output": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, params=request_params)
                data = response.json()
        except Exception as e:
            logger.error(f"[weather_search] request failed: {e}")
            return f"[weather_search] request failed: {e}"

        if data.get("status") != "1":
            return f"[weather_search] failed: {json.dumps(data, ensure_ascii=False)}"

        return json2md(data)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    tool = WeatherSearch()

    result = asyncio.run(
        tool.call(
            {
                "city": "成都",
            }
        )
    )

    print(result)