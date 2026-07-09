# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: POI搜索工具
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
MAX_POI_RESULTS = os.environ.get("MAX_POI_RESULTS", "3")


@register_tool("poi_search", allow_overwrite=True)
class POISearch(BaseTool):
    name = "poi_search"
    description = "按文本搜索地点，返回地址、经纬度、商业信息。"

    parameters = {
        "type": "object",
        "properties": {
            "address": {
                "type": "string",
                "description": "待检索地点文本，例如 春熙路、成都东站、北京故宫。"
            },
            "region": {
                "type": "string",
                "description": "可选，城市级区域，例如 成都市、北京市、上海市。"
            }
        },
        "required": ["address"]
    }

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)

    async def call(self, params: dict, **kwargs) -> str:
        try:
            address = params["address"]
            region = params.get("region", None)
        except Exception:
            return "[POISearch] Invalid request format: Input must be a JSON object containing 'address' field"

        url = "https://restapi.amap.com/v5/place/text"

        request_params = {
            "key": AMAP_MAPS_API_KEY,
            "keywords": address,
            "page_size": MAX_POI_RESULTS,
            "show_fields": "business",
        }

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
            return f"[POISearch] API response error: {msg}"

        pois = result.get("pois")
        if not pois:
            return f"[POISearch] No POI data available for {params}."

        return truncate_text(json2md(pois))


if __name__ == "__main__":
    search = POISearch()

    params = {
        "address": "春熙路",
        "region": "成都市",
    }

    res = asyncio.run(search.call(params))
    print(res)