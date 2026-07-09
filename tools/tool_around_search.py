# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 周边搜工具
# --------------------------------------------

import sys
import os
import asyncio
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from utils.markdown import json2md
from utils.text import truncate_text
from qwen_agent.tools.base import BaseTool, register_tool


AMAP_MAPS_API_KEY = os.environ["AMAP_MAPS_API_KEY"]


@register_tool("around_search", allow_overwrite=True)
class AroundSearch(BaseTool):
    name = "around_search"
    description = "以圆心+半径搜索周边地点。"

    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "中心点经纬度，格式 lng,lat"
            },
            "radius": {
                "type": "integer",
                "description": "半径，单位米，0-50000，默认5000"
            },
            "keyword": {
                "type": "string",
                "description": "可选，单个关键词，例如 餐厅、酒店、景点、停车场"
            },
            "region": {
                "type": "string",
                "description": "可选，城市级区域，例如 成都市、北京市"
            },
        },
        "required": ["location"]
    }

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)

    async def call(self, params: dict, **kwargs) -> str:
        try:
            location = params["location"]
            radius = params.get("radius", 5000)
            keyword = params.get("keyword", None)
            region = params.get("region", None)
        except Exception:
            return "[AroundSearch] Invalid request format: Input must be a JSON object containing 'location' field"

        url = "https://restapi.amap.com/v5/place/around"

        request_params = {
            "key": AMAP_MAPS_API_KEY,
            "location": location,
            "radius": radius,
            "show_fields": "business",
        }

        if keyword:
            request_params["keywords"] = keyword

        if region:
            request_params["region"] = region

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(url, params=request_params)
                response.raise_for_status()
                result = response.json()
            finally:
                await client.aclose()

        if result.get("status") != "1":
            msg = result.get("info", "unknown error")
            return f"[AroundSearch] API response error: {msg}"

        pois = result.get("pois")
        if not pois:
            return f"[AroundSearch] No POI data available for {params}."

        return truncate_text(json2md(pois))


if __name__ == "__main__":
    search = AroundSearch()

    params = {
        "location": "104.077774,30.655544",
        "radius": 1000,
        "keyword": "餐厅",
        "region": "成都市",
    }

    res = asyncio.run(search.call(params))
    print(res)