# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 路线规划工具
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


async def reverse_geocode(location: str):
    """
    逆地理编码：
    经纬度 -> 地址信息

    transit 公交路线规划需要 citycode，
    所以要先根据起点/终点经纬度反查城市编码。
    """
    url = "https://restapi.amap.com/v3/geocode/regeo"

    params = {
        "key": AMAP_MAPS_API_KEY,
        "location": location,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


async def get_citycode(location: str):
    """
    根据经纬度获取 citycode。
    主要给公交路线规划 transit 使用。
    """
    result = await reverse_geocode(location)

    try:
        citycode = result["regeocode"]["addressComponent"]["citycode"]
    except Exception:
        citycode = None

    return citycode


async def driving_direction(
    origin: str,
    destination: str,
    waypoints: str | None = None,
):
    """
    驾车路线规划。
    支持途经点 waypoints，多个途经点用 ; 分隔。
    """
    url = "https://restapi.amap.com/v5/direction/driving"

    params = {
        "key": AMAP_MAPS_API_KEY,
        "origin": origin,
        "destination": destination,
    }

    if waypoints:
        params["waypoints"] = waypoints

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


async def walking_direction(origin: str, destination: str):
    """
    步行路线规划。
    """
    url = "https://restapi.amap.com/v5/direction/walking"

    params = {
        "key": AMAP_MAPS_API_KEY,
        "origin": origin,
        "destination": destination,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


async def bicycling_direction(origin: str, destination: str):
    """
    骑行路线规划。
    """
    url = "https://restapi.amap.com/v5/direction/bicycling"

    params = {
        "key": AMAP_MAPS_API_KEY,
        "origin": origin,
        "destination": destination,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


async def electrobike_direction(origin: str, destination: str):
    """
    电动车路线规划。
    """
    url = "https://restapi.amap.com/v5/direction/electrobike"

    params = {
        "key": AMAP_MAPS_API_KEY,
        "origin": origin,
        "destination": destination,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


async def transit_direction(origin: str, destination: str):
    """
    公交/地铁路线规划。

    高德公交路线规划不只需要经纬度，
    还需要起点城市 city1 和终点城市 city2。
    所以这里先通过 reverse_geocode 获取 citycode。
    """
    url = "https://restapi.amap.com/v5/direction/transit/integrated"

    citycode_origin, citycode_destination = await asyncio.gather(
        get_citycode(origin),
        get_citycode(destination),
    )

    if not citycode_origin:
        return f"City not found for transit origin for {origin} and {destination}."

    if not citycode_destination:
        return f"City not found for transit destination for {origin} and {destination}."

    params = {
        "key": AMAP_MAPS_API_KEY,
        "origin": origin,
        "destination": destination,
        "city1": citycode_origin,
        "city2": citycode_destination,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


@register_tool("route_planning", allow_overwrite=True)
class RoutePlanning(BaseTool):
    name = "route_planning"
    description = "路线规划：驾车/步行/骑行/电动车/公交。"

    parameters = {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "起点经纬度，经度在前，格式 lng,lat",
            },
            "destination": {
                "type": "string",
                "description": "终点经纬度，经度在前，格式 lng,lat",
            },
            "mode": {
                "type": "string",
                "enum": ["driving", "walking", "bicycling", "electrobike", "transit"],
                "description": "路线类型，默认 driving",
            },
            "waypoints": {
                "type": "string",
                "description": "途经点，多个点以 ; 分隔，每点格式 lng,lat",
            },
        },
        "required": ["origin", "destination"],
    }

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)

    async def call(self, params: dict, **kwargs) -> str:
        try:
            origin = params["origin"]
            destination = params["destination"]
            mode = params.get("mode", "driving")
            waypoints = params.get("waypoints")
        except Exception:
            return "[RoutePlanning] Invalid request format: Input must be a JSON object containing 'origin' and 'destination' field"

        if mode == "driving":
            result = await driving_direction(origin, destination, waypoints=waypoints)
        elif mode == "walking":
            result = await walking_direction(origin, destination)
        elif mode == "bicycling":
            result = await bicycling_direction(origin, destination)
        elif mode == "electrobike":
            result = await electrobike_direction(origin, destination)
        elif mode == "transit":
            result = await transit_direction(origin, destination)
        else:
            return f"[RoutePlanning] Unsupported mode: {mode}"

        if not isinstance(result, dict):
            return str(result)

        if result.get("status") != "1":
            msg = result.get("info", "unknown error")
            return f"[RoutePlanning] API response error: {msg}"

        route = result.get("route")
        if not route:
            return f"[RoutePlanning] No route available for {params}."

        return truncate_text(json2md(route))


if __name__ == "__main__":
    search = RoutePlanning()

    params = {
        "origin": "104.077774,30.655544",
        "destination": "104.080091,30.653389",
        "mode": "walking",
    }

    res = asyncio.run(search.call(params))
    print(res)